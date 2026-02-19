import pandas as pd
import configparser
import ast
import json
import re
import uuid
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from dfcx_scrapi.core.sessions import Sessions
from dfcx_scrapi.core.playbooks import Playbooks


# ─── Helpers ──────────────────────────────────────────────────────────────────
def normalize_text(x):
    if x is None:
        return None
    return str(x).strip().lower().replace(" ", "")

def ensure_text(x):
    if x is None:
        return ""
    if not isinstance(x, str):
        x = str(x)
    return x.strip()

def parse_params_maybe_dict(value):
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        s = value.strip()
        if not s:
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
        except Exception:
            pass
    return None

def extract_dob_from_params(params):
    p = parse_params_maybe_dict(params)
    if not isinstance(p, dict):
        return None
    dob = p.get("extracted_dob")
    if not dob or not all(k in dob for k in ("month", "day", "year")):
        return None
    try:
        month = str(int(dob["month"]))
        day   = str(dob["day"]).zfill(2)
        year  = str(dob["year"])
        return f"{month}{day}{year}"
    except Exception:
        return None

def get_headintent(params):
    """Extract headintent from session parameters — this is the detected intent."""
    p = parse_params_maybe_dict(params)
    if not isinstance(p, dict):
        return ""
    val = p.get("headintent", "")
    if isinstance(val, dict):
        return str(val.get("stringValue", "")).strip()
    return str(val).strip() if val else ""


# ─── Single utterance test ────────────────────────────────────────────────────
def test_utterance(sessions_obj, agent_id, utterance, session_id, language_code):
    """
    Calls Sessions.detect_intent without query_params.
    The playbook is reached because the agent_id points to a playbook agent
    and we set current_playbook via the session_id prefix approach.
    headintent is read from the returned session parameters.
    """
    try:
        response    = sessions_obj.detect_intent(
            agent_id=agent_id,
            session_id=session_id,
            text=utterance,
            language_code=language_code,
        )

        qr = response.query_result

        # ── Read parameters directly from proto ───────────────────────────────
        params_dict = {}
        try:
            params_dict = dict(qr.parameters)
        except Exception:
            pass

        # ── Response text ─────────────────────────────────────────────────────
        response_texts = []
        try:
            for msg in qr.response_messages:
                if msg.text and msg.text.text:
                    response_texts.extend(msg.text.text)
        except Exception:
            pass

        # ── headintent = detected intent set by playbook/webhook ──────────────
        detected_intent = get_headintent(params_dict)

        return {
            "detected_intent": detected_intent,
            "parameters":      params_dict,
            "response_text":   " | ".join(response_texts),
            "error":           "",
        }

    except Exception as e:
        return {
            "detected_intent": "",
            "parameters":      {},
            "response_text":   "",
            "error":           str(e),
        }


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    # Step 1: Config
    config = configparser.ConfigParser()
    config.read("config.properties")

    agent_id              = config.get("dialogflow", "agent_id")
    excel_file            = config.get("input", "excel_file")
    sheet_name            = config.getint("input", "sheet_name")
    playbook_display_name = config.get("input", "playbook_display_name")
    Utterances            = config.get("input", "Utterances")
    intent_column         = config.get("input", "intent_column")
    language_code         = config.get("input", "language_code", fallback="en")
    max_workers           = config.getint("input", "max_workers", fallback=10)
    result_file           = config.get("output", "result_file")
    results_excel_file    = (
        config.get("output", "results_excel_file")
        if config.has_option("output", "results_excel_file")
        else "Playbook_Results.xlsx"
    )

    # Step 2: Resolve playbook display name → resource path
    print(f"Resolving playbook: '{playbook_display_name}'...")
    pb_client     = Playbooks(agent_id=agent_id)
    playbooks_map = pb_client.get_playbooks_map(agent_id=agent_id, reverse=True)

    if playbook_display_name not in playbooks_map:
        raise ValueError(
            f"Playbook '{playbook_display_name}' not found.\n"
            f"Available: {list(playbooks_map.keys())}"
        )

    playbook_resource_path = playbooks_map[playbook_display_name]
    playbook_id            = playbook_resource_path.split("/")[-1]
    print(f"Playbook ID: {playbook_id}")

    # Step 3: Sessions object
    sessions_obj = Sessions(agent_id=agent_id)

    # Step 4: Read Excel
    df = pd.read_excel(excel_file, sheet_name=sheet_name, engine="openpyxl")

    for col in [Utterances, intent_column]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in '{excel_file}'.")

    df[Utterances] = (
        df[Utterances].replace({"nan": ""}).fillna("").astype(str).str.strip()
    )
    df = df[df[Utterances] != ""].reset_index(drop=True)

    utterances      = df[Utterances].apply(ensure_text).tolist()
    expected_intent = df[intent_column].astype(str).str.strip().tolist()

    print(f"Loaded {len(utterances)} utterances.")

    # Step 5: Build session IDs with playbook_id prefix
    # This is how dfcx_scrapi routes to a specific playbook —
    # the session ID prefix tells the agent which playbook to use
    session_ids = [
        f"{playbook_id}/{uuid.uuid4()}" for _ in utterances
    ]

    # Step 6: Run in parallel
    raw_results = [None] * len(utterances)

    def run_one(idx):
        utt = utterances[idx]
        sid = session_ids[idx]
        print(f"[{idx+1}/{len(utterances)}] {utt}")
        return idx, test_utterance(
            sessions_obj, agent_id, utt, sid, language_code
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_one, i): i for i in range(len(utterances))}
        for future in as_completed(futures):
            idx, result = future.result()
            raw_results[idx] = result

    # Step 7: Build results DataFrame
    results = pd.DataFrame({
        "utterance":       utterances,
        "expected_intent": expected_intent,
        "detected_intent": [r["detected_intent"] for r in raw_results],
        "response_text":   [r["response_text"]   for r in raw_results],
        "parameters_json": [json.dumps(r["parameters"]) if r["parameters"] else ""
                            for r in raw_results],
        "error":           [r["error"]            for r in raw_results],
    })

    # Step 8: Intent match
    intent_match = (
        results["expected_intent"].apply(normalize_text) ==
        results["detected_intent"].apply(normalize_text)
    )
    results["intent_match"] = intent_match.astype(bool)
    accuracy = float(results["intent_match"].mean() * 100)

    # Step 9: DoB entity extraction
    results["extracted_entity"] = [
        extract_dob_from_params(r["parameters"]) for r in raw_results
    ]
    entity_match = (
        results["extracted_entity"].apply(normalize_text) ==
        results["expected_intent"].apply(normalize_text)
    )
    results["entity_match"] = entity_match.astype(bool)

    # Step 10: Write back to input df
    if "Actual DoB"      not in df.columns: df["Actual DoB"]      = ""
    if "Pass / Fail"     not in df.columns: df["Pass / Fail"]     = ""
    if "Detected Intent" not in df.columns: df["Detected Intent"] = ""

    df["Actual DoB"]      = results["extracted_entity"].astype(str).values
    df["Pass / Fail"]     = np.where(results["entity_match"].values, "Pass", "Fail")
    df["Detected Intent"] = results["detected_intent"].values

    # Step 11: Save
    results.to_csv(result_file, index=False)
    print(f"\nAccuracy: {accuracy:.2f}%")
    print(results[["utterance", "expected_intent", "detected_intent",
                   "intent_match"]].head(10))

    with pd.ExcelWriter(results_excel_file, engine="openpyxl") as writer:
        results.to_excel(writer, sheet_name="All Results", index=False)
        df.to_excel(writer, sheet_name="Annotated Input", index=False)
        pd.DataFrame({
            "Metric": ["Total", "Intent PASS", "Intent FAIL",
                       "Entity PASS", "Entity FAIL", "Accuracy"],
            "Value":  [len(results),
                       int(results["intent_match"].sum()),
                       int((~results["intent_match"]).sum()),
                       int(results["entity_match"].sum()),
                       int((~results["entity_match"]).sum()),
                       f"{accuracy:.2f}%"],
        }).to_excel(writer, sheet_name="Summary", index=False)

        failures = results[~results["intent_match"]]
        if not failures.empty:
            failures.to_excel(writer, sheet_name="Failures", index=False)

        errors = results[results["error"].str.strip() != ""]
        if not errors.empty:
            errors.to_excel(writer, sheet_name="API Errors", index=False)

    print(f"Results saved: {results_excel_file}")


if __name__ == "__main__":
    main()
