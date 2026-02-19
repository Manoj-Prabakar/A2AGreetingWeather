"""
debug_response.py
─────────────────
Runs a single utterance against your playbook agent and dumps the full
raw response so we can find exactly which field holds current_playbook.

Uses the raw Google SDK SessionsClient directly (NOT dfcx_scrapi Sessions)
because Sessions.detect_intent does not accept query_params argument.
"""

import uuid
import json
import configparser
from google.cloud import dialogflowcx_v3beta1 as dialogflow
from google.protobuf.json_format import MessageToDict
from dfcx_scrapi.core.playbooks import Playbooks

# ── Load config ───────────────────────────────────────────────────────────────
config = configparser.ConfigParser()
config.read("config.properties")

agent_id              = config.get("dialogflow", "agent_id")
playbook_display_name = config.get("input", "playbook_display_name")
language_code         = config.get("input", "language_code", fallback="en")

# Parse project_id, location, agent_uuid from full resource path
# Format: projects/<proj>/locations/<loc>/agents/<uuid>
parts      = agent_id.split("/")
project_id = parts[1]
location   = parts[3]
agent_uuid = parts[5]

# ── Test utterance — change this to one from your Excel ───────────────────────
TEST_UTTERANCE = "my date of birth is January 1st 1990"

# ── Resolve playbook display name → resource path ─────────────────────────────
pb_client     = Playbooks(agent_id=agent_id)
playbooks_map = pb_client.get_playbooks_map(agent_id=agent_id, reverse=True)

if playbook_display_name not in playbooks_map:
    print(f"Playbook '{playbook_display_name}' not found!")
    print(f"Available playbooks: {list(playbooks_map.keys())}")
    exit(1)

playbook_resource_path = playbooks_map[playbook_display_name]

print(f"\n{'='*60}")
print(f"Agent    : {agent_id}")
print(f"Playbook : {playbook_display_name}")
print(f"Resource : {playbook_resource_path}")
print(f"Utterance: {TEST_UTTERANCE}")
print(f"{'='*60}\n")

# ── Build raw SDK SessionsClient ──────────────────────────────────────────────
api_endpoint = (
    "dialogflow.googleapis.com:443"
    if location == "global"
    else f"{location}-dialogflow.googleapis.com:443"
)
sdk_client = dialogflow.SessionsClient(
    client_options={"api_endpoint": api_endpoint}
)

session_id   = str(uuid.uuid4())
session_path = sdk_client.session_path(project_id, location, agent_uuid, session_id)

# ── Build and send request ────────────────────────────────────────────────────
request = dialogflow.DetectIntentRequest(
    session=session_path,
    query_input=dialogflow.QueryInput(
        text=dialogflow.TextInput(text=TEST_UTTERANCE),
        language_code=language_code,
    ),
    query_params=dialogflow.QueryParameters(
        current_playbook=playbook_resource_path
    ),
)

response = sdk_client.detect_intent(request=request)

# ── Dump full raw response ────────────────────────────────────────────────────
print("=" * 60)
print("  FULL RAW RESPONSE (MessageToDict)")
print("=" * 60)
raw_dict = MessageToDict(response._pb)
print(json.dumps(raw_dict, indent=2))

# ── Print just queryResult keys ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("  QUERY RESULT TOP-LEVEL KEYS")
print("=" * 60)
qr_raw = raw_dict.get("queryResult", {})
for key, value in qr_raw.items():
    print(f"  {key}: {str(value)[:120]}")

# ── Direct attribute access ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  DIRECT ATTRIBUTE ACCESS on response.query_result")
print("=" * 60)
qr = response.query_result
for attr in ["current_playbook", "response_messages", "parameters", "diagnostic_info"]:
    val = getattr(qr, attr, "NOT FOUND")
    print(f"  qr.{attr}: {str(val)[:150]}")

print("\n" + "=" * 60)
print("  DONE — Share output above so we can fix current_playbook field")
print("=" * 60)
