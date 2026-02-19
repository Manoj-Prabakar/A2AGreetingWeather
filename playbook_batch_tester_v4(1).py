import pandas as pd
import configparser
import ast
import json
import re
import uuid
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─── dfcx_scrapi imports (same library as your flow workstream) ───────────────
# Flow workstream used:  dfcx_scrapi.core.conversation.DialogflowConversation
# Playbook workstream:   dfcx_scrapi.core.sessions.Sessions
#
# Why Sessions instead of DialogflowConversation?
#   - DialogflowConversation.run_intent_detection() is built for FLOW agents.
#     It reads detected_intent + confidence from NLU match results.
#   - For PLAYBOOK agents the NLU match is absent; the LLM routes the turn.
#     Sessions.detect_intent() returns the raw QueryResult protobuf so we can
#     read current_playbook, parameters, and response_messages ourselves.
from dfcx_scrapi.core.sessions import Sessions
from dfcx_scrapi.core.playbooks import Playbooks
from google.cloud.dialogflowcx_v3beta1.types import QueryParameters


# ─── Helpers (kept identical to your flow workstream) ────────────────────────
def parse_params_maybe_dict(value):
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        s = value.strip()
        if s == "":
            return None
        try:
            d = json.loads(s)
            if isinstance(d, dict):
                return d
        except Exception:
            pass
        try:
            d = ast.literal_eval(s)
            if isinstance(d, dict):
                return d
        except (ValueError, SyntaxError):
            return None
    return None


def normalize_text(x):
    if x is None:
        return None
    return str(x).strip().lower().replace(" ", "")


def normalize_entity_value(v, digits_only=True):
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        v = "".join(str(x) for x in v)
    s = str(v)
    s = re.sub(r"\s+", "", s)
    if digits_only:
        s = re.sub(r"\D+", "", s)
    return s


def ensure_text(x):
    if x is None:
        return ""
    if not isinstance(x, str):
        x = str(x)
    return x.strip()


# ─── DoB extraction (same logic as your flow workstream) ─────────────────────
def extract_dob_from_params(params: dict):
    """
    Extract DoB from session parameters returned by the playbook agent.
    Looks for 'extracted_dob' with keys 'month', 'day', 'year'.
    Returns formatted string M{DD}{YYYY} or None.
    """
    if not isinstance(params, dict):
        return None
    dob = params.get("extracted_dob")
    if not dob or not all(k in dob for k in ("month", "day", "year")):
        return None
    try:
        month = str(int(dob["month"]))      # no leading zero on month
        day   = str(dob["day"]).zfill(2)    # 2-digit day
        year  = str(dob["year"])            # 4-digit year
        return f"{month}{day}{year}"
    except Exception:
        return None


# ─── Single-utterance detect_intent via dfcx_scrapi Sessions ─────────────────
def run_single_detect_intent(
    sessions_obj, agent_id, utterance, playbook_resource_path, language_code="en"
):
    """
    Call Sessions.detect_intent() for one utterance and parse the playbook response.

    From debug output we know:
      - qr.current_playbook does NOT exist as a direct attribute
      - The playbook name lives at:
          queryResult -> generativeInfo -> actionTracingInfo -> name
        which is a resource path like:
          projects/.../agents/.../playbooks/<playbook_id>
      - We extract the last segment as the playbook name
    """
    from google.protobuf.json_format import MessageToDict

    session_id = str(uuid.uuid4())   # fresh session per utterance

    try:
        # ── Build QueryParameters with initial playbook ───────────────────────
        query_params = QueryParameters(
            current_playbook=playbook_resource_path
        )

        # ── Call detect_intent ────────────────────────────────────────────────
        response = sessions_obj.detect_intent(
            agent_id=agent_id,
            session_id=session_id,
            text=utterance,
            language_code=language_code,
            query_params=query_params,
        )

        # ── Convert full response to dict for parsing ─────────────────────────
        raw_dict = MessageToDict(response._pb)
        qr_raw   = raw_dict.get("queryResult", {})

        # ── Extract current_playbook from generativeInfo.actionTracingInfo ────
        # From debug output, the playbook resource path is nested here:
        #   generativeInfo -> actionTracingInfo -> name
        # e.g. projects/.../agents/.../playbooks/<playbook_id>
        current_playbook_resource = (
            qr_raw
            .get("generativeInfo", {})
            .get("actionTracingInfo", {})
            .get("name", "")
        )
        # Extract just the display segment — last part of the resource path
        current_playbook_name = (
            current_playbook_resource.split("/")[-1]
            if current_playbook_resource else ""
        )

        # ── Response text ─────────────────────────────────────────────────────
        response_texts = []
        for msg in qr_raw.get("responseMessages", []):
            texts = msg.get("text", {}).get("text", [])
            response_texts.extend(texts)

        # ── Parameters ────────────────────────────────────────────────────────
        qr          = response.query_result
        raw_params  = getattr(qr, "parameters", None)
        params_dict = dict(raw_params) if raw_params else {}

        return {
            "current_playbook":          current_playbook_name,
            "current_playbook_resource": current_playbook_resource,
            "response_text":             " | ".join(response_texts),
            "parameters":                params_dict if isinstance(params_dict, dict) else {},
            "error":                     "",
        }

    except Exception as e:
        return {
            "current_playbook":          "",
            "current_playbook_resource": "",
            "response_text":             "",
            "parameters":                {},
            "error":                     str(e),
        }


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    # ── Step 1: Load config (same config.properties as your flow workstream) ──
    config = configparser.ConfigParser()
    config.read("config.properties")

    # ── Step 2: Read config values ────────────────────────────────────────────
    agent_id          = config.get("dialogflow", "agent_id")
    creds_path        = config.get("dialogflow", "creds_path", fallback=None)

    excel_file        = config.get("input", "excel_file")
    sheet_name        = config.getint("input", "sheet_name")

    # playbook_display_name: the display name of the playbook to start the session at.
    # Equivalent to flow_display_name + page_display_name in your flow workstream.
    # The script resolves this to a full resource path automatically using Playbooks class.
    playbook_display_name = config.get("input", "playbook_display_name")

    Utterances        = config.get("input", "Utterances")
    intent_column     = config.get("input", "intent_column")
    language_code     = config.get("input", "language_code", fallback="en")
    max_workers       = config.getint("input", "max_workers", fallback=10)

    result_file        = config.get("output", "result_file")
    results_excel_file = (
        config.get("output", "results_excel_file")
        if config.has_option("output", "results_excel_file")
        else "Playbook_Results.xlsx"
    )

    # ── Step 3: Create Sessions object (replaces DialogflowConversation) ──────
    if creds_path:
        sessions_obj = Sessions(creds_path=creds_path, agent_id=agent_id)
    else:
        sessions_obj = Sessions(agent_id=agent_id)

    # ── Step 3.1: Resolve playbook display name → full resource path ──────────
    # This is the playbook equivalent of flow_display_name in your flow workstream.
    # Playbooks.get_playbooks_map() returns {display_name: resource_path} so we
    # can look up the resource path by the human-readable display name from config.
    print(f"Resolving playbook display name: '{playbook_display_name}'...")
    if creds_path:
        pb_client = Playbooks(creds_path=creds_path, agent_id=agent_id)
    else:
        pb_client = Playbooks(agent_id=agent_id)

    # reverse=True → {display_name: resource_path}
    playbooks_map = pb_client.get_playbooks_map(agent_id=agent_id, reverse=True)

    if playbook_display_name not in playbooks_map:
        available = list(playbooks_map.keys())
        raise ValueError(
            f"Playbook '{playbook_display_name}' not found in agent.\n"
            f"Available playbooks: {available}"
        )

    playbook_resource_path = playbooks_map[playbook_display_name]
    print(f"Resolved playbook resource path: {playbook_resource_path}")

    # ── Step 4: Read input Excel (identical to your flow workstream) ──────────
    df = pd.read_excel(excel_file, sheet_name=sheet_name, engine="openpyxl")

    for col in [Utterances, intent_column]:
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found in sheet '{sheet_name}' of '{excel_file}'."
            )

    df[Utterances] = (
        df[Utterances]
        .replace({"nan": ""})
        .fillna("")
        .astype(str)
        .str.strip()
    )
    df = df[df[Utterances] != ""].reset_index(drop=True)

    utterances         = df[Utterances].apply(ensure_text).tolist()
    expected_playbooks = df[intent_column].astype(str).str.strip().tolist()

    print(f"Loaded {len(utterances)} utterances. Starting playbook detection...")

    # ── Step 5: Run detect_intent in parallel (mirrors run_intent_detection) ──
    # Your flow workstream used conversation.run_intent_detection(test_set, 100, 10)
    # which internally uses ThreadPoolExecutor. We replicate that here directly.
    raw_results = [None] * len(utterances)

    def test_one(idx):
        utt = utterances[idx]
        print(f"[{idx + 1}/{len(utterances)}] Testing: '{utt}'")
        return idx, run_single_detect_intent(
            sessions_obj, agent_id, utt, playbook_resource_path, language_code
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(test_one, i): i for i in range(len(utterances))}
        for future in as_completed(futures):
            idx, result = future.result()
            raw_results[idx] = result

    # ── Step 6: Build results DataFrame ──────────────────────────────────────
    results = pd.DataFrame({
        "utterance":                [r["utterance"] if "utterance" in r else utterances[i]
                                     for i, r in enumerate(raw_results)],
        "expected_playbook":        expected_playbooks,
        "current_playbook":         [r["current_playbook"]          for r in raw_results],
        "current_playbook_resource":[r["current_playbook_resource"] for r in raw_results],
        "response_text":            [r["response_text"]             for r in raw_results],
        "parameters_json":          [json.dumps(r["parameters"]) if r["parameters"] else ""
                                     for r in raw_results],
        "error":                    [r["error"]                     for r in raw_results],
    })
    results["utterance"] = utterances   # ensure correct order

    # ── Step 7: Playbook match (replaces intent_match from flow workstream) ───
    playbook_string_match = (
        results["expected_playbook"].apply(normalize_text) ==
        results["current_playbook"].apply(normalize_text)
    )
    # Playbooks have NO confidence score (LLM-based). We skip the conf >= 0.3
    # gate that your flow workstream used and rely purely on name match.
    results["playbook_match"] = playbook_string_match.astype(bool)
    results["confidence"]     = "N/A"   # kept for schema parity with flow results

    accuracy = float(results["playbook_match"].mean() * 100)

    # ── Step 7.1: Entity extraction (same DoB logic as your flow workstream) ──
    params_list = [r["parameters"] for r in raw_results]

    results["extracted_entity"] = [
        extract_dob_from_params(p) for p in params_list
    ]

    # Entity match: compare extracted DoB vs expected value in intent_column
    entity_string_match = (
        results["extracted_entity"].apply(normalize_text) ==
        results["expected_playbook"].apply(normalize_text)
    )
    results["entity_match"] = entity_string_match.astype(bool)

    # ── Step 7.2: Write back to original input DataFrame ─────────────────────
    # Mirrors your flow workstream's df["Actual DoB"] / df["Pass / Fail"] write-back
    if "Actual DoB" not in df.columns:
        df["Actual DoB"] = ""
    if "Pass / Fail" not in df.columns:
        df["Pass / Fail"] = ""
    if "Matched Playbook" not in df.columns:
        df["Matched Playbook"] = ""

    df["Actual DoB"]       = results["extracted_entity"].astype(str).values
    df["Pass / Fail"]      = np.where(results["entity_match"].values, "Pass", "Fail")
    df["Matched Playbook"] = results["current_playbook"].values

    # ── Step 8: Save CSV + Excel (same as your flow workstream) ───────────────
    results.to_csv(result_file, index=False)
    print(f"\nTest completed.")
    print(f"Playbook match accuracy : {accuracy:.2f}%")
    print(f"Full results CSV saved  : {result_file}")

    # Print results1-style debug view (mirrors your flow workstream's df_one print)
    print("\nPrinting results DataFrame (first 10 rows):")
    print(results[["utterance", "expected_playbook", "current_playbook",
                   "playbook_match", "extracted_entity", "entity_match"]].head(10))

    # Multi-sheet Excel report
    with pd.ExcelWriter(results_excel_file, engine="openpyxl") as writer:
        results.to_excel(writer, sheet_name="All Results", index=False)
        df.to_excel(writer, sheet_name="Annotated Input", index=False)

        summary = pd.DataFrame({
            "Metric": [
                "Total Utterances",
                "Playbook PASS",
                "Playbook FAIL",
                "Entity PASS",
                "Entity FAIL",
                "API Errors",
                "Playbook Match Accuracy (%)",
            ],
            "Value": [
                len(results),
                int(results["playbook_match"].sum()),
                int((~results["playbook_match"]).sum()),
                int(results["entity_match"].sum()),
                int((~results["entity_match"]).sum()),
                int(results["error"].str.strip().ne("").sum()),
                f"{accuracy:.2f}%",
            ],
        })
        summary.to_excel(writer, sheet_name="Summary", index=False)

        failures = results[~results["playbook_match"]]
        if not failures.empty:
            failures.to_excel(writer, sheet_name="Failures", index=False)

        errors = results[results["error"].str.strip() != ""]
        if not errors.empty:
            errors.to_excel(writer, sheet_name="API Errors", index=False)

    print(f"Results Excel written   : {results_excel_file}")


if __name__ == "__main__":
    main()
