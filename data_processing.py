import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake

def load_repository_and_initialize_database(root_dir, deeplake_account_name):
    # Load repository
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

    return db, embeddings
