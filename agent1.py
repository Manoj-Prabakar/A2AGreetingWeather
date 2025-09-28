import asyncio
import json
from datetime import datetime
from typing import List
import httpx
import nest_asyncio
from a2a.client import A2ACardResolver
from a2a.types import (
    AgentCard,
    MessageSendParams,
    SendMessageRequest,
    SendMessageResponse,
    SendMessageSuccessResponse,
    Task,
)
from dotenv import load_dotenv
from google.adk import Agent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from .remote_agent_connection import RemoteAgentConnections

load_dotenv()
nest_asyncio.apply()


class GreetingAgent:
    """The Greeting agent that interacts with the Weather agent."""

    def __init__(self):
        self.remote_agent_connections: dict[str, RemoteAgentConnections] = {}
        self.cards: dict[str, AgentCard] = {}
        self.agents: str = ""
        self._agent = self.create_agent()
        self._user_id = "greeting_agent"
        self._runner = Runner(
            app_name=self._agent.name,
            agent=self._agent,
            artifact_service=InMemoryArtifactService(),
            session_service=InMemorySessionService(),
            memory_service=InMemoryMemoryService(),
        )

    async def _async_init_components(self, remote_agent_addresses: List[str]):
        async with httpx.AsyncClient(timeout=30) as client:
            for address in remote_agent_addresses:
                card_resolver = A2ACardResolver(client, address)
                try:
                    card = await card_resolver.get_agent_card()
                    remote_connection = RemoteAgentConnections(
                        agent_card=card, agent_url=address
                    )
                    self.remote_agent_connections[card.name] = remote_connection
                    self.cards[card.name] = card
                except httpx.ConnectError as e:
                    print(f"ERROR: Failed to get agent card from {address}: {e}")
                except Exception as e:
                    print(f"ERROR: Failed to initialize connection for {address}: {e}")
        agent_info = [
            json.dumps({"name": card.name, "description": card.description})
            for card in self.cards.values()
        ]
        print("agent_info:", agent_info)
        self.agents = "\n".join(agent_info) if agent_info else "No weather agent found"

    @classmethod
    async def create(cls, remote_agent_addresses: List[str]):
        instance = cls()
        await instance._async_init_components(remote_agent_addresses)
        return instance

    def create_agent(self) -> Agent:
        return Agent(
            model="gemini-2.5-flash",
            name="Greeting_Agent",
            instruction=self.root_instruction,
            description="This agent sends greetings and requests weather information from the weather agent.",
            tools=[self.send_message],
        )

    def root_instruction(self, context: ReadonlyContext) -> str:
        return f"""
        **Role:** You are the Greeting Agent. 
        - Your job is to greet the user politely.  
        - If the user’s query is about the weather (e.g., mentions temperature, forecast, rain, snow, sunny, cloudy, storm, climate, etc.), 
          you must immediately transfer control to the **Weather Agent**.  
        - Do not answer weather-related questions yourself. Just hand them over.  
        - For all other queries, respond as the Greeting Agent with a friendly greeting.  

        **Today's Date (YYYY-MM-DD):** {datetime.now().strftime("%Y-%m-%d")}
        <Available Agents>
        {self.agents}
        </Available Agents>
        """

    async def send_message(self, agent_name: str, city: str, tool_context: ToolContext):
        if agent_name not in self.remote_agent_connections:
            raise ValueError(f"Agent {agent_name} not found")
        client = self.remote_agent_connections[agent_name]
        if not client:
            raise ValueError(f"Client not available for {agent_name}")
        message_id = tool_context.state.get("message_id", "greet-weather-1")
        payload = {
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": f"Hello! Can you tell me the weather in {city} today?"}],
                "messageId": message_id,
            },
        }
        message_request = SendMessageRequest(
            id=message_id, params=MessageSendParams.model_validate(payload)
        )
        send_response: SendMessageResponse = await client.send_message(message_request)
        print("send_response", send_response)
        if not isinstance(send_response.root, SendMessageSuccessResponse) or not isinstance(send_response.root.result, Task):
            print("Received a non-success or non-task response. Cannot proceed.")
            return
        response_content = send_response.root.model_dump_json(exclude_none=True)
        json_content = json.loads(response_content)
        resp = []
        if json_content.get("result", {}).get("artifacts"):
            for artifact in json_content["result"]["artifacts"]:
                if artifact.get("parts"):
                    resp.extend(artifact["parts"])
        return resp


def _get_initialized_greeting_agent_sync():
    """Synchronously creates and initializes the GreetingAgent."""

    async def _async_main():
        weather_agent_urls = [
            "http://localhost:10002",  # Weather Agent
        ]
        print("initializing greeting agent")
        greeting_agent_instance = await GreetingAgent.create(
            remote_agent_addresses=weather_agent_urls
        )
        print("GreetingAgent initialized")
        return greeting_agent_instance.create_agent()

    try:
        return asyncio.run(_async_main())
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" in str(e):
            print(
                f"Warning: Could not initialize GreetingAgent with asyncio.run(): {e}. "
                "This can happen if an event loop is already running (e.g., in Jupyter). "
                "Consider initializing GreetingAgent within an async function in your application."
            )
        else:
            raise


root_agent = _get_initialized_greeting_agent_sync()
