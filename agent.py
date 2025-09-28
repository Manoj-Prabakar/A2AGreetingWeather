from fastapi import FastAPI, Request
import uvicorn
from datetime import date
from google.adk.agents import LlmAgent

app = FastAPI()

# Predefined weather data for demonstration
CITY_WEATHER = {
    "London": "Cloudy, 18°C",
    "New York": "Sunny, 25°C",
    "Paris": "Rainy, 16°C",
    "Tokyo": "Clear, 22°C",
    "Sydney": "Windy, 20°C"
}

def get_weather(city: str) -> str:
    """
    Returns today's weather for a given city.
    Args:
        city: The name of the city to check weather for.
    Returns:
        A string describing the weather for today in the city.
    """
    today = date.today().strftime("%Y-%m-%d")
    weather = CITY_WEATHER.get(city.title())
    if weather:
        return f"Weather in {city.title()} on {today}: {weather}"
    else:
        return f"Sorry, I don't have weather data for {city}."

def create_agent() -> LlmAgent:
    """Constructs the ADK agent for Weather."""
    return LlmAgent(
        model="gemini-2.5-flash",
        name="Weather_Agent",
        instruction="""
            **Role:** You are a weather information agent. 
            Your sole responsibility is to provide today's weather for predefined cities: London, New York, Paris, Tokyo, Sydney.
            **Core Directives:**
            *   **Provide Weather:** Use the `get_weather` tool to answer questions about the weather in supported cities. 
            *   **Polite and Concise:** Always be polite and to the point in your responses.
            *   **Stick to Your Role:** Do not engage in any conversation outside of weather information. If asked other questions, politely state that you can only help with weather.
        """,
        tools=[get_weather],
    )

@app.post("/message")
async def receive_message(request: Request):
    data = await request.json()
    message = data.get("message", {})
    text = ""
    if isinstance(message, dict):
        parts = message.get("parts", [])
        if parts and isinstance(parts[0], dict):
            text = parts[0].get("text", "")
    elif isinstance(message, str):
        text = message
    # Simple parsing to extract city name
    city = None
    for city_name in CITY_WEATHER.keys():
        if city_name.lower() in text.lower():
            city = city_name
            break
    today = date.today().strftime("%Y-%m-%d")
    if city:
        response = f"Weather in {city} on {today}: {CITY_WEATHER[city]}"
    else:
        response = "Sorry, I don't have weather data for that city."
    print(f"Received message: {text}")
    return {"response": response}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=10002)
