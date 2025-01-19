# from langchain_community.chat_models import ChatOllama
import sys
import argparse

from langchain_ollama import ChatOllama
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate


def create_chatbot():
    """
    Creates a chatbot using Ollama and LangChain.

    Returns:
        ConversationChain: A chain for interacting with the chatbot.
    """

    # Initialize the Ollama model
    llm = ChatOllama(model="llama3.2", temperature=0.7) 

    # Create a memory to store conversation history
    memory = ConversationBufferMemory()

    # Create a conversation chain
    conversation_chain = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )

    return conversation_chain


def main():

    chatbot = create_chatbot()

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        output = chatbot.run(input=user_input)
        print("Chatbot:", output)


def get_llm():
    return ChatOllama(
        model="llama3.2", 
        temperature=0.7
        )


def method_invocation(system_role):
    llm = get_llm()
    user_input = input("you: ")
    message = [
        ("system", system_role),
        ("human", user_input)        
    ]
    resp = llm.invoke(message)
    print(f"returned obj: {type(resp)}, {resp}")
    print(f"AI: {resp.content}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--invocation",
        action="store_true",
        help="invoke AIMessage with prompt"
        )
    parser.add_argument(
        "--chaining",
        action="store_true",
        help="initiate chat session with prompt"
        )
    args = parser.parse_args()

    if not args.invocation and not args.chaining:
        parser.print_help()
        sys.exit(1)

    system_role = input("system role: ")
    print(f"len: {len(system_role)}, system role: {system_role}")

    if not system_role:
        print("since no role defined, using default as translator")
        system_role = "you are an experienced python programmer"
        print(f"system role: {system_role}")

    # main()
    if args.invocation:
        method_invocation(system_role=system_role)


# 0xAA55
