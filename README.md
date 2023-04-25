# Talking to Myself

Talking to Myself (talking-to-myself) is a conversational command line application that uses a language model to interact with a collection of documents. The app uses the Langchain library and the OpenAI API to enable intelligent conversations about the ingested documents. You can use this application to explore your document collection by asking questions and getting relevant answers.

## Prerequisites

Before you start using Talking to Myself, you need to have the following:

- Python 3.6 or higher installed on your system
- An OpenAI API key

## Installation

1. Clone this repository to your local machine:
```bash
git clone https://github.com/jonoberg/talking-to-myself.git
cd talking-to-myself
```

2. Create a virtual environment and activate it:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Configuration

This project requires your OpenAI API key and the path to directory containing the documents you want to ingest. If you don't want to specify a directory to ingest your documents from you can instead place your documents inside of the `/ingest` folder and leave the `root_dir` value blank. There are three ways to configure these settings:

1. JSON file: Create a JSON file named `secrets.json` in the root directory of the project with the following format:

```json
{
  "openai_api_key": "your_openai_api_key_here",
  "root_dir": "path/to/your/document/directory"
}
```

2. DOTENV file: Alternatively, you can create a `.env` file in the root directory of the project with the following format:
```
OPENAI_API_KEY=your_openai_api_key_here
ROOT_DIR=path/to/your/document/directory
```

Don't forget to install the `python-dotenv` package:
```bash
pip install python-dotenv
```


3. Environment variables: If you prefer not to use a secrets.json or .env file, you can set environment variables named OPENAI_API_KEY and ROOT_DIR with the appropriate values:
```bash
export OPENAI_API_KEY="your_openai_api_key_here"
export ROOT_DIR="path/to/your/document/directory"
```

The config.py file will first try to load the settings from the secrets.json file. If the file does not exist, it will try to load the settings from the .env file. If the settings are still not found, it will try to load them from environment variables.

## Usage

1. Process the documents in your collection by running the following command:
```bash
python app.py --process-data
```

This command ingests, embeds, and saves the documents to disk. You only need to run this command once, unless you add or modify documents in your collection.

2. Start the app by running:
```bash
python app.py
```

You can now interact with your document collection by asking questions and receiving relevant answers.

## Contributing

Feel free to contribute to this project by submitting pull requests, reporting bugs, or suggesting new features.

## License

This project is licensed under the MIT License.