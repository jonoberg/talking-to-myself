# settings.py

from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

def init_settings(retriever, llm, chain):
    retriever.search_kwargs['distance_metric'] = 'cos'
    retriever.search_kwargs['fetch_k'] = 10
    retriever.search_kwargs['maximal_marginal_relevance'] = True
    retriever.search_kwargs['k'] = 10

    llm.model_name = 'gpt-3.5-turbo'
    llm.temperature = 0
    llm.streaming = True
    llm.callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm.verbose = True

    chain.max_tokens_limit = 4096
    chain.return_source_documents = True

    return retriever, llm, chain