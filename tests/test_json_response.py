import asyncio
from pydantic import BaseModel

from llm_client.clients.openai_client import OpenAIClient


class Step(BaseModel):
    explanation: str
    output: str


class MathReasoning(BaseModel):
    steps: list[Step]
    final_answer: str


client = OpenAIClient()


def test_json_response_response_api():
    messages = [
        {
            "role": "system",
            "content": "You are a helpful math tutor. Guide the user through the solution step by step.",
        },
        {"role": "user", "content": "how can I solve 8x + 7 = -23"},
    ]
    result = client.get_json_response(
        messages=messages,
        model="doubao-seed-1.6",
        output_type=MathReasoning,
        extra_body={"thinking": {"type": "disabled"}},
        use_response=True,
    )
    print(result)
    print(client.cost)
    print(client.token_usage)
    assert isinstance(result, MathReasoning)


def test_json_response_response_completion():
    messages = [
        {
            "role": "system",
            "content": """You are a helpful math tutor. Guide the user through the solution step by step.
You must answer in JSON format like:
{
    "steps": [
        {
            "explanation": "explanation",
            "output": "output"
        }
    ]
    "final_answer": "final_answer"
}
""",
        },
        {"role": "user", "content": "how can I solve 8x + 7 = -23"},
    ]
    result = client.get_json_response(
        messages=messages,
        model="deepseek",
        output_type=MathReasoning,
        extra_body={"thinking": {"type": "disabled"}},
        use_response=False,
    )
    print(result)
    print(client.cost)
    print(client.token_usage)
    assert isinstance(result, MathReasoning)


async def test_json_response_response_api_async():
    messages = [
        {
            "role": "system",
            "content": "You are a helpful math tutor. Guide the user through the solution step by step.",
        },
        {"role": "user", "content": "how can I solve 8x + 7 = -23"},
    ]
    result = await client.get_json_response_async(
        messages=messages,
        model="doubao-seed-1.6",
        output_type=MathReasoning,
        extra_body={"thinking": {"type": "disabled"}},
        use_response=True,
    )
    print(result)
    print(client.cost)
    print(client.token_usage)
    assert isinstance(result, MathReasoning)


if __name__ == "__main__":
    # test_json_response_response_api()
    asyncio.run(test_json_response_response_api_async())
