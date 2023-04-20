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

# User input
questions = [
    "Is there a chain the implements a chat interface on top of an LLM, will retrieve relevant documents from a vectorstore and will cite it's sources?",
    # "What classes are derived from the Chain class?",
    # "What classes and functions in the ./langchain/utilities/ forlder are not covered by unit tests?",
    # "What one improvement do you propose in code in relation to the class herarchy for the Chain class?",
] 

#Init chat history
chat_history = []

# Query LLM with User Input
# Append user input and LLM response to chat history
# Print user input and LLM reponse to stdout
for question in questions:  
    result = qa({"question": question, "chat_history": chat_history})
    chat_history.append((question, result['answer']))
    print(f"-> **Question**: {question} \n")
    print(f"**Answer**: {result['answer']} \n")