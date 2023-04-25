def get_user_input(prompt):
    return input(prompt)

def chat_loop(qa):
    # Init chat history
    chat_history = []

    print("Welcome to talking-to-myself! Feel free to ask your notes any questions you need help with.")
    print("Type 'exit' or 'quit' to end the conversation.\n")

    while True:
        question = get_user_input("\n-> You: ")

        if question.lower() in ('exit', 'quit'):
            print("Goodbye!")
            break

        print(f"-> AI: ", end='', flush=True)
        result = qa({"question": question, "chat_history": chat_history, "return_source_documents": True})
        chat_history.append((question, result['answer']))

        # Print source documents
        if 'source_documents' in result:
            print("\nSource documents:")
            for i, doc in enumerate(result['source_documents'], start=1):
                print(f"Document {i} metadata: {doc.metadata}")
                print(f"{i}. {doc.metadata['source']}")
