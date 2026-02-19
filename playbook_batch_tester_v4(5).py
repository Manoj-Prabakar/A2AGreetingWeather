import pandas as pd
import configparser
import ast
import json
import re
import numpy as np
from dfcx_scrapi.core.conversation import DialogflowConversation
from dfcx_scrapi.core.playbooks import Playbooks


# ─── Helpers (identical to your flow workstream) ──────────────────────────────
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


def ensure_text(x):
    if x is None:
        return ""
    if not isinstance(x, str):
        x = str(x)
    return x.strip()


# ─── DoB extraction (identical to your flow workstream) ───────────────────────
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


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    # Step 1: Load config
    config = configparser.ConfigParser()
    config.read("config.properties")

    # Step 2: Read config values
    agent_id              = config.get("dialogflow", "agent_id")
    excel_file            = config.get("input", "excel_file")
    sheet_name            = config.getint("input", "sheet_name")
    playbook_display_name = config.get("input", "playbook_display_name")
    Utterances            = config.get("input", "Utterances")
    intent_column         = config.get("input", "intent_column")
    result_file           = config.get("output", "result_file")
    results_excel_file    = (
        config.get("output", "results_excel_file")
        if config.has_option("output", "results_excel_file")
        else "Playbook_Results.xlsx"
    )

    # Step 3: Resolve playbook display name → full resource path
    # Used in test_set as current_playbook — equivalent of flow_display_name
    print(f"Resolving playbook: '{playbook_display_name}'...")
    pb_client     = Playbooks(agent_id=agent_id)
    playbooks_map = pb_client.get_playbooks_map(agent_id=agent_id, reverse=True)

    if playbook_display_name not in playbooks_map:
        raise ValueError(
            f"Playbook '{playbook_display_name}' not found.\n"
            f"Available: {list(playbooks_map.keys())}"
        )

    playbook_resource_path = playbooks_map[playbook_display_name]
    print(f"Resolved: {playbook_resource_path}")

    # Step 4: Create DialogflowConversation — same as flow workstream
    conversation = DialogflowConversation(agent_id=agent_id)

    # Step 5: Read input Excel
    df = pd.read_excel(excel_file, sheet_name=sheet_name, engine="openpyxl")

    for col in [Utterances, intent_column]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in '{excel_file}'.")

    df[Utterances] = (
        df[Utterances]
        .replace({"nan": ""})
        .fillna("")
        .astype(str)
        .str.strip()
    )
    df = df[df[Utterances] != ""].reset_index(drop=True)

    # Step 6: Build test_set — same structure as flow workstream but with
    # current_playbook instead of flow_display_name + page_display_name
    test_set = pd.DataFrame({
        "current_playbook": playbook_resource_path,   # ← playbook equivalent of flow_display_name
        "utterance":        df[Utterances].apply(ensure_text),
    })
    test_set["utterance"] = test_set["utterance"].str.slice(0, 4000)

    # Step 7: Run intent detection — same as flow workstream
    print(f"Running detect_intent on {len(test_set)} utterances...")
    results = conversation.run_intent_detection(test_set, 100, 10)

    print("Printing results DataFrame:")
    print(results)

    # Step 8: Extract headintent from parameters — this is the detected intent
    # Your playbook/webhook sets headintent inside session parameters
    if "parameters" not in results.columns and "parameters_set" in results.columns:
        results["parameters"] = results["parameters_set"]

    results["expected_intent"] = df[intent_column].values
    results = results.reset_index(drop=True)

    # Extract headintent from parameters dict
    def get_headintent(params):
        p = parse_params_maybe_dict(params)
        if not isinstance(p, dict):
            return ""
        val = p.get("headintent", "")
        if isinstance(val, dict):
            return val.get("stringValue", str(val)).strip()
        return str(val).strip() if val else ""

    results["detected_intent"] = results["parameters"].apply(get_headintent)

    # Step 9: Intent match
    intent_match = (
        results["expected_intent"].apply(normalize_text) ==
        results["detected_intent"].apply(normalize_text)
    )
    results["intent_match"] = intent_match.astype(bool)
    accuracy = float(results["intent_match"].mean() * 100)

    # Step 10: Entity extraction (DoB) — identical to flow workstream
    results["extracted_entity"] = results["parameters"].apply(extract_dob_from_params)

    entity_match = (
        results["extracted_entity"].apply(normalize_text) ==
        results["expected_intent"].apply(normalize_text)
    )
    results["entity_match"] = entity_match.astype(bool)

    # Step 11: Write back to input DataFrame
    if "Actual DoB"       not in df.columns: df["Actual DoB"]       = ""
    if "Pass / Fail"      not in df.columns: df["Pass / Fail"]      = ""
    if "Detected Intent"  not in df.columns: df["Detected Intent"]  = ""

    df["Actual DoB"]      = results["extracted_entity"].astype(str).values
    df["Pass / Fail"]     = np.where(results["entity_match"].values, "Pass", "Fail")
    df["Detected Intent"] = results["detected_intent"].values

    # Step 12: Save outputs
    results.to_csv(result_file, index=False)
    print(f"\nTest completed. Accuracy: {accuracy:.2f}%")

    with pd.ExcelWriter(results_excel_file, engine="openpyxl") as writer:
        results.to_excel(writer, sheet_name="All Results", index=False)
        df.to_excel(writer, sheet_name="Annotated Input", index=False)

        summary = pd.DataFrame({
            "Metric": ["Total Utterances", "Intent PASS", "Intent FAIL",
                       "Entity PASS", "Entity FAIL", "Accuracy (%)"],
            "Value":  [len(results),
                       int(results["intent_match"].sum()),
                       int((~results["intent_match"]).sum()),
                       int(results["entity_match"].sum()),
                       int((~results["entity_match"]).sum()),
                       f"{accuracy:.2f}%"],
        })
        summary.to_excel(writer, sheet_name="Summary", index=False)

        failures = results[~results["intent_match"]]
        if not failures.empty:
            failures.to_excel(writer, sheet_name="Failures", index=False)

        errors = results[results.get("error", pd.Series([""] * len(results))).str.strip() != ""]
        if not errors.empty:
            errors.to_excel(writer, sheet_name="API Errors", index=False)

    print(f"Results Excel written: {results_excel_file}")


if __name__ == "__main__":
    main()
