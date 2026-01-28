llm-client
==========

A lightweight client wrapper for OpenAI-compatible chat APIs with streaming, tool calls, and simple cost tracking.

Setup
-----

1. Copy `src/llm_client/clients/openai_client.py` into your project
2. Create a configuration file based on `src/llm_client/conf/llm.example.yaml`
3. Update the config with your API endpoint and key

Usage
-----

```python
from llm_client import OpenAIClient

client = OpenAIClient()
messages = [{"role": "user", "content": "Say hi!"}]
print(client.send_messages(messages, model="deepseek"))
```
