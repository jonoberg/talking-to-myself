from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

def get_user_input(prompt):
    return input(prompt)

def chat_loop(qa):
    # Init chat history
    chat_history = []

    print("Welcome to codechat! Feel free to ask your code repository any questions you need help with.")
    print("Type 'exit' or 'quit' to end the conversation.\n")

    while True:
        question = get_user_input("-> You: ")

        if question.lower() in ('exit', 'quit'):
            print("Goodbye!")
            break

        result = qa({"question": question, "chat_history": chat_history})
        chat_history.append((question, result['answer']))
        print(f"-> AI: {result['answer']}\n")
