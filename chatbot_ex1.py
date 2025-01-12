from langchain.chat_models import ChatOllama
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

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
    """
    Main function to interact with the chatbot.
    """

    chatbot = create_chatbot()

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        output = chatbot.run(input=user_input)
        print("Chatbot:", output)

if __name__ == "__main__":
    main()

