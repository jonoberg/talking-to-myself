import os
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.environ.get("OPENAI_API_KEY")
activeloop_api_key = os.environ.get("ACTIVELOOP_TOKEN")
deeplake_account_name = os.environ.get("DEEPLAKE_ACCOUNT_NAME")

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

def get_user_input(prompt):
    return input(prompt)

# Load repository
root_dir = './ingest'
docs = []
for dirpath, dirnames, filenames in os.walk(root_dir):
    for file in filenames:
        if file.endswith('.py') and '/.venv/' not in dirpath:
            try: 
                loader = TextLoader(os.path.join(dirpath, file), encoding='utf-8')
                docs.extend(loader.load_and_split())
            except Exception as e: 
                pass

# Split repository documents into text chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(docs)

# Init embeddings object
embeddings = OpenAIEmbeddings()

# Send text chunks to OpenAI Embeddings API
# Send text chunks and embeddings to Deep Lake
db = DeepLake.from_documents(texts, embeddings, dataset_path=f"hub://{deeplake_account_name}/langchain-code")

# Init database and config
db = DeepLake(dataset_path=f"hub://{deeplake_account_name}/langchain-code", read_only=True, embedding_function=embeddings)
retriever = db.as_retriever()
retriever.search_kwargs['distance_metric'] = 'cos'
retriever.search_kwargs['fetch_k'] = 10
retriever.search_kwargs['maximal_marginal_relevance'] = True
retriever.search_kwargs['k'] = 10

# Init and config LLM
model = ChatOpenAI(model='gpt-3.5-turbo') # 'ada' 'gpt-3.5-turbo' 'gpt-4',
qa = ConversationalRetrievalChain.from_llm(model,retriever=retriever)

#Init chat history
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