from dotenv import load_dotenv

from client.llm_client import LLMClient
import asyncio

load_dotenv()

async def main():
    client = LLMClient()
    messages = [{
        'role': 'user',
        'content': 'Hello, how are you?'
    }]
    async for event in client.chat_completion(messages, False):
        print(event)
    print("Done")

asyncio.run(main())