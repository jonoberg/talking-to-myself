import json
import os
from dotenv import load_dotenv

def load_configuration():
    openai_api_key = None
    root_dir = None

    # Try to load from secrets.json
    try:
        with open('secrets.json', 'r') as f:
            config = json.load(f)
            openai_api_key = config.get('openai_api_key')
            root_dir = config.get('root_dir')
    except FileNotFoundError:
        pass

    # If not found in secrets.json, try loading from .env
    if not openai_api_key or not root_dir:
        load_dotenv()
        openai_api_key = os.environ.get('OPENAI_API_KEY', openai_api_key)
        root_dir = os.environ.get('ROOT_DIR', root_dir)

    # If not found in .env, load from user set environment variables
    if not openai_api_key:
        openai_api_key = os.environ.get('OPENAI_API_KEY')

    if not root_dir:
        root_dir = os.environ.get('ROOT_DIR')

    if not openai_api_key:
        raise ValueError("Missing OpenAI API key in secrets.json, .env, or environment variables")

    return openai_api_key, root_dir
