from llm_client import OpenAIClient


def main():
    messages = [{"role": "user", "content": "Hello, how are you?"}]
    client = OpenAIClient()
    print(client.get_str_response(messages))


if __name__ == "__main__":
    main()
