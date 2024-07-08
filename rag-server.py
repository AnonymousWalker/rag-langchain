# ! pip install langchain_community tiktoken langchain-openai langchainhub chromadb langchain flask
import os
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = ''
os.environ['OPENAI_API_KEY'] = ''

# ### Vector Store DB & Retriever
db_path = "D:/misc/rag/rag-db-v2"

#### INDEXING ####
import bs4
from langchain_community.document_loaders import TextLoader

# Index
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from database import create_db

# vectorstore = Chroma.from_documents(documents=splits, persist_directory=db_path, embedding=OpenAIEmbeddings())
# vectorstore = create_db(db_path)
vectorstore = Chroma(persist_directory=db_path, embedding_function=OpenAIEmbeddings())

retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={'k': 5, 'lambda_mult': 0.25})

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

llm = ChatOpenAI(temperature=0)

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

#   -------------- EXECUTION ------------
def send_prompt_rag_plain(question: str):
    # Prompt
    template = """Answer the question given following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    docs = retriever.get_relevant_documents(question)

    return {
        'response': rag_chain.invoke(question),
        'context': list(map(lambda doc: doc.page_content, docs))
    }


def send_prompt_llm(prompt: str):
    # test llm
    messages = [
        ("system", "Answer the user prompt."),
        ("user", prompt),
    ]
    return llm.invoke(messages).content

def extract_keywords(prompt: str):
    messages = [
        ("system", "Extract the keywords from the user prompt and order them by the terminology. The response should contain only comma-separated strings."),
        ("user", prompt)
    ]
    raw_keywords = llm.invoke(messages).content.split(',')
    keywords = [s.strip() for s in raw_keywords]
    return keywords


# ### SERVER
from flask import Flask, request, jsonify
from flask_cors import CORS
from glossary import create_glossary

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
dictionary = create_glossary(r'D:\misc\rag\data\en_tw')

@app.route('/rag', methods=['GET'])
def get_prompt():
    prompt = request.args.get('prompt', default='', type=str)
    keywords = extract_keywords(prompt)
    
    tw_dict = {}
    for k in keywords:
        found = dictionary.get(k.lower(), '')
        if found != '':
            tw_dict[k] = found

    response = {
        'rag-response:' : send_prompt_rag_plain(prompt),
        'llm-response': send_prompt_llm(prompt),
        'keywords': keywords,
        'tw': tw_dict
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
