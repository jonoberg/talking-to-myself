# talking-to-myself

import sys
import argparse
import pickle

from config import load_configuration
from settings import init_settings
from data_processing import ingest_data
from chat import chat_loop
from prompt_templates import get_prompt_templates

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

# Load configuration
openai_api_key, root_dir = load_configuration()

# Parse CLI for flags
parser = argparse.ArgumentParser()
parser.add_argument('--process-data', action='store_true', help='Run the data processing step before starting the chat loop')
args = parser.parse_args()

def main():
    with open("db.pkl", "rb") as f:
        db = pickle.load(f)

    CONDENSE_QUESTION_PROMPT, QA_PROMPT = get_prompt_templates()

    retriever = db.as_retriever()

    llm = ChatOpenAI()

    chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        qa_prompt=QA_PROMPT,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT
        )

    retriever, llm, chain = init_settings(retriever, llm, chain)

    # Start chat loop
    chat_loop(chain)

if args.process_data:
    if not root_dir:
        root_dir = './ingest'
    ingest_data(root_dir)
    print("Documents successfully ingested, embedded and saved to disk. Run the app.py file again but without the '--process-data' flag to chat with your docs!")
else:
    main()