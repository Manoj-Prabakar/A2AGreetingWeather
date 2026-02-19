"""
debug_response.py
─────────────────
Run this FIRST with a single utterance to see the full raw response
structure returned by your playbook agent.

This will tell us exactly which key holds the current_playbook name
so we can fix the main batch tester.

Usage:
    python debug_response.py
"""

import uuid
import json
import configparser
from google.protobuf.json_format import MessageToDict
from dfcx_scrapi.core.sessions import Sessions
from dfcx_scrapi.core.playbooks import Playbooks
from google.cloud.dialogflowcx_v3beta1.types import QueryParameters

# ── Load config ───────────────────────────────────────────────────────────────
config = configparser.ConfigParser()
config.read("config.properties")

agent_id              = config.get("dialogflow", "agent_id")
creds_path            = config.get("dialogflow", "creds_path", fallback=None)
playbook_display_name = config.get("input", "playbook_display_name")
language_code         = config.get("input", "language_code", fallback="en")

# ── Test utterance — change this to any utterance from your Excel ─────────────
TEST_UTTERANCE = "my date of birth is January 1st 1990"

# ── Create Sessions client ────────────────────────────────────────────────────
sessions_obj = Sessions(agent_id=agent_id)

# ── Resolve playbook display name → resource path ─────────────────────────────
pb_client    = Playbooks(agent_id=agent_id)
playbooks_map          = pb_client.get_playbooks_map(agent_id=agent_id, reverse=True)
playbook_resource_path = playbooks_map[playbook_display_name]

print(f"\n{'='*60}")
print(f"Agent       : {agent_id}")
print(f"Playbook    : {playbook_display_name}")
print(f"Resource    : {playbook_resource_path}")
print(f"Utterance   : {TEST_UTTERANCE}")
print(f"{'='*60}\n")

# ── Run detect_intent ─────────────────────────────────────────────────────────
query_params = QueryParameters(current_playbook=playbook_resource_path)

response = sessions_obj.detect_intent(
    agent_id=agent_id,
    session_id=str(uuid.uuid4()),
    text=TEST_UTTERANCE,
    language_code=language_code,
    query_params=query_params,
)

# ── Print FULL raw response as JSON ───────────────────────────────────────────
# This shows every field returned — we use this to find where current_playbook lives
print("=" * 60)
print("  FULL RAW RESPONSE (MessageToDict)")
print("=" * 60)
raw_dict = MessageToDict(response._pb)
print(json.dumps(raw_dict, indent=2))

# ── Print queryResult keys specifically ───────────────────────────────────────
print("\n" + "=" * 60)
print("  QUERY RESULT TOP-LEVEL KEYS")
print("=" * 60)
qr_raw = raw_dict.get("queryResult", {})
for key, value in qr_raw.items():
    # Print key and a preview of the value
    preview = str(value)[:120]
    print(f"  {key}: {preview}")

# ── Try collect_playbook_responses ────────────────────────────────────────────
print("\n" + "=" * 60)
print("  collect_playbook_responses() OUTPUT")
print("=" * 60)
try:
    playbook_data = sessions_obj.collect_playbook_responses(response)
    print(json.dumps(playbook_data, indent=2, default=str))
except Exception as e:
    print(f"  ERROR calling collect_playbook_responses: {e}")

# ── Try direct attribute access on query_result ───────────────────────────────
print("\n" + "=" * 60)
print("  DIRECT ATTRIBUTE ACCESS on response.query_result")
print("=" * 60)
qr = response.query_result
attrs = [
    "current_playbook",
    "current_flow",
    "match",
    "intent",
    "parameters",
    "response_messages",
    "diagnostic_info",
]
for attr in attrs:
    val = getattr(qr, attr, "ATTRIBUTE NOT FOUND")
    print(f"  qr.{attr}: {str(val)[:120]}")

print("\n" + "=" * 60)
print("  DONE — Share the output above to identify the correct field")
print("=" * 60)
