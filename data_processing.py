import os
import pickle
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

def ingest_data(root_dir):
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
    db = FAISS.from_documents(texts, embeddings)

    # Save vectorstore
    with open("db.pkl", "wb") as f:
        pickle.dump(db, f)

# def data_processing(root_dir, deeplake_account_name):
#     # Load repository and initialize database
#     db, embeddings = load_repository_and_initialize_database(root_dir, deeplake_account_name)
#     return db, embeddings
