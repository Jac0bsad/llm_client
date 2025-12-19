from llm_client import OpenAIClient


def main():
    messages = [{"role": "user", "content": "Hello, how are you?"}]
    client = OpenAIClient()
    print(client.send_messages(messages))


if __name__ == "__main__":
    main()
