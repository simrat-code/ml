
import asyncio
# from urllib import response
import ollama
import sys
# from ollama import AsyncClient

def sync_example():
    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": "Tell me something about elephant in a room",
            },
        ],
    )

    print(response["message"]["content"])


async def chat():
    message = {
        "role": "user",
        "content": "tell me an interesting fact about sigma male"
    }

    async for part in await ollama.AsyncClient().chat(
        model="llama3.2",
        messages=[message],
        stream=True
    ):
        print(part["message"]["content"], end="", flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "sync":
        print(f"executing sync function, it wll take some time...")
        sync_example()
    elif len(sys.argv) == 2 and sys.argv[1] == "async":
        print(f"executing async funtion...\n")
        asyncio.run(chat())
    else:
        print("valid options:")
        print(f"{sys.argv[0]} sync|async")
