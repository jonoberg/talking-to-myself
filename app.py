import sys
import argparse
import pickle
from config import load_configuration
from data_processing import ingest_data
from chat import chat_loop
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationalRetrievalChain

# Parse CLI for flags
parser = argparse.ArgumentParser()
parser.add_argument('--process-data', action='store_true', help='Run the data processing step before starting the chat loop')
args = parser.parse_args()

# Load configuration
openai_api_key, activeloop_api_key, deeplake_account_name = load_configuration()

if args.process_data:
    root_dir = './ingest'
    ingest_data(root_dir)
    print("Documents successfully ingested, embedded and saved to disk. Run the app.py file again but without the '--process-data' flag to chat with your docs!")
    sys.exit()
else:
    with open("db.pkl", "rb") as f:
        db = pickle.load(f)

# Init database and config
retriever = db.as_retriever()
retriever.search_kwargs['distance_metric'] = 'cos'
retriever.search_kwargs['fetch_k'] = 10
retriever.search_kwargs['maximal_marginal_relevance'] = True
retriever.search_kwargs['k'] = 10

# Init and config LLM
model = ChatOpenAI(
    model='gpt-3.5-turbo',
    temperature=0,
    streaming=True, 
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), 
    verbose=True
    )
qa = ConversationalRetrievalChain.from_llm(
    model,
    retriever=retriever
    )

# Start chat loop
chat_loop(qa)