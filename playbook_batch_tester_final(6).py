import pandas as pd
import configparser
import ast
import json
import uuid
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from openpyxl import load_workbook
from openpyxl.styles import PatternFill, Font, Alignment

# Raw Google SDK — only correct way to pass current_playbook via QueryParameters
from google.cloud import dialogflowcx_v3beta1 as dialogflow
import time
import os

# dfcx_scrapi only used for resolving playbook display name → resource path
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

def extract_param(params_dict, key):
    """Extract a string value from parameters dict by key."""
    if not isinstance(params_dict, dict):
        return ""
    val = params_dict.get(key, "")
    if isinstance(val, dict):
        return str(val.get("stringValue", "")).strip()
    return str(val).strip() if val else ""


# ─── Single utterance test ────────────────────────────────────────────────────
def test_utterance(sdk_client, session_path, utterance, language_code):
    # Retry up to 5 times on quota/token errors with exponential backoff
    for attempt in range(5):
        try:
            request = dialogflow.DetectIntentRequest(
                session=session_path,
                query_input=dialogflow.QueryInput(
                    text=dialogflow.TextInput(text=utterance),
                    language_code=language_code,
                ),
                query_params=dialogflow.QueryParameters(
                    current_playbook=playbook_resource_path
                ),
            )
            response = sdk_client.detect_intent(request=request)
            qr       = response.query_result
            break  # success — exit retry loop
        except Exception as e:
            err_str = str(e).lower()
            if any(x in err_str for x in ["429", "quota", "resource exhausted", "rate"]):
                wait = (2 ** attempt) * 30  # 30s, 60s, 120s, 240s, 480s
                print(f"  [RATE LIMIT] Attempt {attempt+1}/5 — waiting {wait}s before retry...")
                time.sleep(wait)
                if attempt == 4:
                    return {"params_dict": {}, "response_text": "", "error": str(e)}
            else:
                return {"params_dict": {}, "response_text": "", "error": str(e)}
    try:
        pass  # qr already set above

        # ── Read parameters directly from proto MapComposite ──────────────────
        params_dict = {}
        try:
            for key, val in qr.parameters.items():
                kind = val.WhichOneof("kind")
                if kind == "string_value":
                    params_dict[key] = val.string_value
                elif kind == "number_value":
                    params_dict[key] = val.number_value
                elif kind == "bool_value":
                    params_dict[key] = val.bool_value
                elif kind == "struct_value":
                    params_dict[key] = {
                        k: v.string_value for k, v in val.struct_value.fields.items()
                    }
                elif kind == "list_value":
                    params_dict[key] = [v.string_value for v in val.list_value.values]
                else:
                    params_dict[key] = str(val)
        except Exception as proto_err:
            print(f"[WARN] Proto iteration failed: {proto_err} — trying MessageToDict fallback")
            try:
                from google.protobuf.json_format import MessageToDict as M2D
                raw = M2D(response._pb)
                params_dict = raw.get("queryResult", {}).get("parameters", {})
                print(f"[WARN] MessageToDict params: {params_dict}")
            except Exception as dict_err:
                print(f"[ERROR] MessageToDict also failed: {dict_err}")

        # ── Response text ─────────────────────────────────────────────────────
        response_texts = []
        try:
            for msg in qr.response_messages:
                if msg.text and msg.text.text:
                    response_texts.extend(msg.text.text)
        except Exception:
            pass

        return {
            "params_dict":   params_dict,
            "response_text": " | ".join(response_texts),
            "error":         "",
        }

    except Exception as e:
        return {
            "params_dict":   {},
            "response_text": "",
            "error":         str(e),
        }


# ─── Process one sheet ────────────────────────────────────────────────────────
def process_sheet(df, sdk_client, project_id, location, agent_uuid,
                  utterances_col, head_intent_col, sub_intent_col,
                  language_code, max_workers, sheet_name):

    df[utterances_col] = (
        df[utterances_col].replace({"nan": ""}).fillna("").astype(str).str.strip()
    )
    df = df[df[utterances_col] != ""].reset_index(drop=True)

    utterances          = df[utterances_col].apply(ensure_text).tolist()
    expected_head_intent = df[head_intent_col].astype(str).str.strip().tolist()
    expected_sub_intent  = df[sub_intent_col].astype(str).str.strip().tolist()

    print(f"\n[Sheet: {sheet_name}] Processing {len(utterances)} utterances...")

    # Build unique session path per utterance
    session_paths = [
        sdk_client.session_path(project_id, location, agent_uuid, str(uuid.uuid4()))
        for _ in utterances
    ]

    raw_results = [None] * len(utterances)

    def run_one(idx):
        print(f"  [{idx+1}/{len(utterances)}] {utterances[idx]}")
        result = test_utterance(
            sdk_client, session_paths[idx], utterances[idx], language_code
        )
        # Print first utterance params immediately for verification
        if idx == 0:
            print("\n" + "="*60)
            print(f"  FIRST UTTERANCE CHECK — Sheet: {sheet_name}")
            print("="*60)
            print(f"  Utterance       : {utterances[0]}")
            print(f"  Agent Response  : {result['response_text'] or 'EMPTY'}")
            print(f"  Raw param keys  : {list(result['params_dict'].keys())}")
            print(f"  Parameters      : {json.dumps(result['params_dict'], indent=4, default=str)}")
            print(f"  headintent      : {result['params_dict'].get('headintent', 'NOT FOUND')}")
            print(f"  subintent       : {result['params_dict'].get('subintent', 'NOT FOUND')}")
            # Also try common casing variations
            for k in result['params_dict'].keys():
                if 'intent' in k.lower():
                    print(f"  [intent key found] '{k}' = '{result['params_dict'][k]}'")
            print(f"  Error           : {result['error'] or 'None'}")
            print("="*60 + "\n")
        return idx, result

    checkpoint_file = f"checkpoint_{sheet_name.replace(' ', '_')}.json"

    # Load existing checkpoint if present — resume from where we left off
    completed = set()
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file) as f:
            saved = json.load(f)
        for item in saved:
            raw_results[item["idx"]] = item["result"]
            completed.add(item["idx"])
        print(f"  [RESUME] Loaded {len(completed)} completed utterances from checkpoint.")

    remaining = [i for i in range(len(utterances)) if i not in completed]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_one, i): i for i in remaining}
        for future in as_completed(futures):
            idx, result = future.result()
            raw_results[idx] = result
            completed.add(idx)
            # Save checkpoint every 100 utterances
            if len(completed) % 100 == 0:
                checkpoint_data = [
                    {"idx": i, "result": raw_results[i]}
                    for i in range(len(utterances)) if raw_results[i] is not None
                ]
                with open(checkpoint_file, "w") as f:
                    json.dump(checkpoint_data, f, default=str)
                print(f"  [CHECKPOINT] Saved progress: {len(completed)}/{len(utterances)}")

    # Remove checkpoint file after successful completion
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print(f"  [DONE] Checkpoint removed.")

    # Build output rows
    rows = []
    for idx, r in enumerate(raw_results):
        params           = r["params_dict"]
        detected_head    = extract_param(params, "headintent")
        detected_sub     = extract_param(params, "subintent")
        head_match       = normalize_text(expected_head_intent[idx]) == normalize_text(detected_head)
        sub_match        = normalize_text(expected_sub_intent[idx])  == normalize_text(detected_sub)

        rows.append({
            "utterance":             utterances[idx],
            "expected_head_intent":  expected_head_intent[idx],
            "detected_head_intent":  detected_head,
            "head_intent_match":     head_match,
            "expected_sub_intent":   expected_sub_intent[idx],
            "detected_sub_intent":   detected_sub,
            "sub_intent_match":      sub_match,
            "error":                 r["error"],
        })

    result_df = pd.DataFrame(rows)

    # Calculate accuracy
    total            = len(result_df)
    head_pass        = int(result_df["head_intent_match"].sum())
    sub_pass         = int(result_df["sub_intent_match"].sum())
    head_accuracy    = round(head_pass / total * 100, 2) if total > 0 else 0
    sub_accuracy     = round(sub_pass  / total * 100, 2) if total > 0 else 0

    # Append accuracy rows after last data row
    accuracy_rows = pd.DataFrame([
        {
            "utterance":            "HEAD INTENT ACCURACY",
            "expected_head_intent": "",
            "detected_head_intent": "",
            "head_intent_match":    f"{head_accuracy}%",
            "expected_sub_intent":  "",
            "detected_sub_intent":  "",
            "sub_intent_match":     "",
            "error":                "",
        },
        {
            "utterance":            "SUB INTENT ACCURACY",
            "expected_head_intent": "",
            "detected_head_intent": "",
            "head_intent_match":    "",
            "expected_sub_intent":  "",
            "detected_sub_intent":  "",
            "sub_intent_match":     f"{sub_accuracy}%",
            "error":                "",
        },
    ])

    final_df = pd.concat([result_df, accuracy_rows], ignore_index=True)

    print(f"  [Sheet: {sheet_name}] Head Accuracy: {head_accuracy}% | Sub Accuracy: {sub_accuracy}%")
    return final_df


# ─── Apply Excel formatting ───────────────────────────────────────────────────
def format_sheet(ws):
    TRUE_FILL  = PatternFill("solid", fgColor="C6EFCE")
    FALSE_FILL = PatternFill("solid", fgColor="FFC7CE")
    HDR_FILL   = PatternFill("solid", fgColor="2E4057")
    HDR_FONT   = Font(color="FFFFFF", bold=True)
    ACC_FILL   = PatternFill("solid", fgColor="FFEB9C")
    ACC_FONT   = Font(bold=True)

    # Style header row
    for cell in ws[1]:
        cell.fill      = HDR_FILL
        cell.font      = HDR_FONT
        cell.alignment = Alignment(horizontal="center", wrap_text=True)
    ws.row_dimensions[1].height = 25
    ws.freeze_panes = "A2"

    # Find head_intent_match and sub_intent_match column indices
    head_col = sub_col = None
    for cell in ws[1]:
        if cell.value == "head_intent_match": head_col = cell.column
        if cell.value == "sub_intent_match":  sub_col  = cell.column

    # Style data rows
    for row in ws.iter_rows(min_row=2):
        utterance_val = str(row[0].value or "")
        is_accuracy   = "ACCURACY" in utterance_val.upper()

        for cell in row:
            cell.alignment = Alignment(vertical="center", wrap_text=True)
            if is_accuracy:
                cell.fill = ACC_FILL
                cell.font = ACC_FONT
                continue
            # Color True/False in match columns
            if head_col and cell.column == head_col:
                if cell.value is True:
                    cell.fill = TRUE_FILL
                    cell.font = Font(bold=True, color="276221")
                elif cell.value is False:
                    cell.fill = FALSE_FILL
                    cell.font = Font(bold=True, color="9C0006")
            if sub_col and cell.column == sub_col:
                if cell.value is True:
                    cell.fill = TRUE_FILL
                    cell.font = Font(bold=True, color="276221")
                elif cell.value is False:
                    cell.fill = FALSE_FILL
                    cell.font = Font(bold=True, color="9C0006")

    # Set column widths
    col_widths = [45, 25, 25, 18, 25, 25, 18, 40, 30]
    for i, width in enumerate(col_widths, 1):
        from openpyxl.utils import get_column_letter
        ws.column_dimensions[get_column_letter(i)].width = width


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    global playbook_resource_path

    # Step 1: Config
    config = configparser.ConfigParser()
    config.read("config.properties")

    agent_id              = config.get("dialogflow", "agent_id")
    excel_file            = config.get("input", "excel_file")
    playbook_display_name = config.get("input", "playbook_display_name")
    utterances_col        = config.get("input", "Utterances")
    head_intent_col       = config.get("input", "head_intent_column")
    sub_intent_col        = config.get("input", "sub_intent_column")
    language_code         = config.get("input", "language_code",   fallback="en")
    max_workers           = config.getint("input", "max_workers",  fallback=10)
    results_excel_file    = (
        config.get("output", "results_excel_file")
        if config.has_option("output", "results_excel_file")
        else "Playbook_Results.xlsx"
    )

    # Step 2: Parse project_id, location, agent_uuid
    parts      = agent_id.split("/")
    project_id = parts[1]
    location   = parts[3]
    agent_uuid = parts[5]

    # Step 3: Resolve playbook display name → resource path
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

    # Step 4: Build raw SDK client
    api_endpoint = (
        "dialogflow.googleapis.com:443"
        if location == "global"
        else f"{location}-dialogflow.googleapis.com:443"
    )
    sdk_client = dialogflow.SessionsClient(
        client_options={"api_endpoint": api_endpoint}
    )

    # Step 5: Read all sheet names from input Excel
    xl              = pd.ExcelFile(excel_file, engine="openpyxl")
    all_sheets      = xl.sheet_names
    start_sheet     = config.getint("input", "start_sheet", fallback=0)
    end_sheet       = config.getint("input", "end_sheet",   fallback=len(all_sheets) - 1)
    sheet_names     = all_sheets[start_sheet : end_sheet + 1]
    print(f"Total sheets: {len(all_sheets)} | Processing sheets {start_sheet} to {end_sheet}: {sheet_names}")

    # Step 6: Process each sheet and write to output Excel
    with pd.ExcelWriter(results_excel_file, engine="openpyxl") as writer:
        for sheet in sheet_names:
            print(f"\nProcessing sheet: '{sheet}'...")
            df = pd.read_excel(excel_file, sheet_name=sheet, engine="openpyxl")

            # Validate required columns
            missing = [c for c in [utterances_col, head_intent_col, sub_intent_col]
                       if c not in df.columns]
            if missing:
                print(f"  Skipping sheet '{sheet}' — missing columns: {missing}")
                continue

            final_df = process_sheet(
                df, sdk_client, project_id, location, agent_uuid,
                utterances_col, head_intent_col, sub_intent_col,
                language_code, max_workers, sheet
            )

            # Write to same sheet name as input
            final_df.to_excel(writer, sheet_name=sheet, index=False)

    # Step 7: Apply formatting to all sheets
    wb = load_workbook(results_excel_file)
    for sheet in wb.sheetnames:
        format_sheet(wb[sheet])
    wb.save(results_excel_file)

    print(f"\nAll sheets processed. Results saved: {results_excel_file}")


if __name__ == "__main__":
    main()
