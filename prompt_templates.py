# prompt_templates.py

from langchain.prompts.prompt import PromptTemplate

def get_prompt_templates():
    _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
    You can assume the follow-up request is about the personal notes of the requestor.

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""
    condense_question_prompt = PromptTemplate.from_template(_template)

    template = """You are an AI assistant for answering questions about the personal notes of the user sending the request. Provide a conversational answer. If there is any context provided as part of the request assume that these documents are from the personal notes of the requestor.
    If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
    If the request is not related to the personal note documents that were sent along or there are no documents sent along, politely inform the requestor that either their personal notes don't aren't able to assist in answering the question or that there weren't any notes provided. Then let the requestor know that you are happy to try to answer the question with your general knowledge if they would like.
    Question: {question}
    =========
    {context}
    =========
    Answer in Markdown:"""
    qa_prompt = PromptTemplate(
        template=template, 
        input_variables=["question", "context"])

    return condense_question_prompt, qa_prompt
