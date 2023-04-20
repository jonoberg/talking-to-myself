import argparse
from config import load_configuration
from data_processing import data_processing, load_repository_and_initialize_database
from chat import chat_loop
from langchain.vectorstores import DeepLake
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

# Parse CLI for flags
parser = argparse.ArgumentParser()
parser.add_argument('--process-data', action='store_true', help='Run the data processing step before starting the chat loop')
args = parser.parse_args()

# Load configuration
openai_api_key, activeloop_api_key, deeplake_account_name = load_configuration()

if args.process_data:
    root_dir = './ingest'
    db, embeddings = data_processing(root_dir, deeplake_account_name)
else:
    embeddings = OpenAIEmbeddings()
    db = DeepLake(dataset_path=f"hub://{deeplake_account_name}/langchain-code", read_only=True, embedding_function=embeddings)

# Init database and config
retriever = db.as_retriever()
retriever.search_kwargs['distance_metric'] = 'cos'
retriever.search_kwargs['fetch_k'] = 10
retriever.search_kwargs['maximal_marginal_relevance'] = True
retriever.search_kwargs['k'] = 10

# Init and config LLM
model = ChatOpenAI(model='gpt-3.5-turbo') # 'ada' 'gpt-3.5-turbo' 'gpt-4',
qa = ConversationalRetrievalChain.from_llm(model, retriever=retriever)

# Start chat loop
chat_loop(qa)