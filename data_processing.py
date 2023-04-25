import os
import pickle
import sys
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

def ingest_data(root_dir):
    # Load documents
    docs = []
    print(f"Checking {root_dir} for documents...")
    for dirpath, dirnames, filenames in os.walk(root_dir):
        print(f"Current directory path: {dirpath}")
        print(f"Subdirectories: {dirnames}")
        print(f"Files: {filenames}\n")
        for file in filenames:
            if file.endswith('.md'):
                try: 
                    loader = UnstructuredMarkdownLoader(os.path.join(dirpath, file))
                    docs.extend(loader.load())
                except Exception as e: 
                    pass
    if len(docs) == 0:
        print(f"No documents found in {root_dir}. Exiting...")
        sys.exit()
    else:
        print(f"Loaded {len(docs)} documents")

    # Split repository documents into text chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(docs)
    print(f"Generated {len(texts)} text chunks")

    # Init embeddings object
    embeddings = OpenAIEmbeddings()

    # Send text chunks to OpenAI Embeddings API
    # Send text chunks and embeddings to Deep Lake
    db = FAISS.from_documents(texts, embeddings)

    # Save vectorstore
    with open("db.pkl", "wb") as f:
        pickle.dump(db, f)
