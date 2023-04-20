from config import load_configuration
from data_processing import load_repository_and_initialize_database
from chat import chat_loop
from langchain.vectorstores import DeepLake
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

# Load configuration
openai_api_key, activeloop_api_key, deeplake_account_name = load_configuration()

# Load repository and initialize database
root_dir = './ingest'
db, embeddings = load_repository_and_initialize_database(root_dir, deeplake_account_name)

# Init database and config
db = DeepLake(dataset_path=f"hub://{deeplake_account_name}/langchain-code", read_only=True, embedding_function=embeddings)
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