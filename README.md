llm-client
==========

A lightweight client wrapper for OpenAI-compatible chat APIs with streaming, tool calls, and simple cost tracking.

Install
-------

- From a git repository:

```bash
pip install git+https://github.com/username/common-package.git
```

- From a local checkout (editable):

```bash
pip install -e .
```

Configuration
-------------

Copy the example config and update with your endpoint and key:

```bash
cp src/llm_client/conf/openai_llms.example.yaml src/llm_client/conf/openai_llms.yaml
```

CLI
---

After install, a console command `llm-client` is available:

```bash
llm-client "Hello! Summarize yourself in one sentence."
```

Python Usage
------------

```python
from llm_client import OpenAIClient

client = OpenAIClient()
messages = [{"role": "user", "content": "Say hi!"}]
print(client.send_messages(messages, model="deepseek"))
```
