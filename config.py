import os
from dotenv import load_dotenv

def load_configuration():
    load_dotenv()
    
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    root_dir = os.environ.get("ROOT_DIR")
    
    return openai_api_key, root_dir
