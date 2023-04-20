import os
from dotenv import load_dotenv

def load_configuration():
    load_dotenv()
    
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    activeloop_api_key = os.environ.get("ACTIVELOOP_TOKEN")
    deeplake_account_name = os.environ.get("DEEPLAKE_ACCOUNT_NAME")
    
    return openai_api_key, activeloop_api_key, deeplake_account_name
