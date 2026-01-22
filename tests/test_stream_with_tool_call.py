import asyncio
from llm_client.clients.openai_client import OpenAIClient


async def get_weather(location: str) -> str:
    """
    Get weather information for a specified location
    Args:
        location: Location name
    Returns:
        Weather information string
    """
    # Test return fixed information
    return "Today's weather is sunny, temperature is 20 degrees"


async def get_room_temperature() -> float:
    """
    Get current room temperature
    Returns:
        Room temperature value
    """
    return 20.0


async def call_tool_function(tool_name: str, tool_args: dict):
    if tool_name == "get_weather":
        return await get_weather(tool_args["location"])
    elif tool_name == "get_room_temperature":
        return await get_room_temperature()


async def test_stream_with_tool_call():
    client = OpenAIClient()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather information for a specified location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "Location name"}
                    },
                    "required": ["location"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_room_temperature",
                "description": "Get current room temperature",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        },
    ]
    messages = [
        {
            "role": "user",
            "content": "Please get the current weather information for Beijing and the current room temperature",
        }
    ]
    async for chunk in client.send_messages_stream_with_tool_call(
        messages=messages,
        tools=tools,
        call_tool_func=call_tool_function,
    ):
        print(chunk)


if __name__ == "__main__":
    asyncio.run(test_stream_with_tool_call())
