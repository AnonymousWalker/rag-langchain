import os
import json

config = {}
if os.path.exists('config.json'):
    with open('config.json', 'r') as conf:
        config = json.load(conf)
else:
    config['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
    config['DATA_SOURCE_DIR'] = os.getenv('DATA_SOURCE_DIR')
    config['DB_PATH'] = os.getenv('DB_PATH')

# Const values
OPENAI_KEY = config['OPENAI_API_KEY']
DB_PATH = config["DB_PATH"]
DATA_SOURCE_DIR = config["DATA_SOURCE_DIR"]

#### INDEXING ####
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from database import create_db

embedding = OpenAIEmbeddings(api_key=OPENAI_KEY)
if os.path.exists(DB_PATH):
    print(f"Database found at {DB_PATH}")
    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embedding)
else:
    print(f"Constructing database at {DB_PATH}")
    vectorstore = create_db(DB_PATH, DATA_SOURCE_DIR, embedding)

retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={'k': 3, 'lambda_mult': 0.5 })

from langchain_openai import OpenAI, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_KEY)

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

#   -------------- EXECUTION ------------
def send_prompt_rag_plain(question: str, system_prompt: str):
    # Prompt
    # template = """Answer the question given following context:
    template = system_prompt + """\nAnswer the question and base on the context if relevant.
    
    Context:
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
        'context': format_docs(docs)
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
        ("system", """
Extract the keywords from the user prompt. 
The response should be a json object with two fields - a list of keywords and a language code of the prompt as `language_code`.
Prompt:
"""),
        ("user", prompt)
    ]
    response = llm.invoke(messages).content
    response_obj = json.loads(response)
    raw_keywords = response_obj["keywords"]
    language = str(response_obj["language_code"])
    keywords = [str(s).strip() for s in raw_keywords]
    
    return keywords, language

# =========== EXPERIMENTAL =========
def prepend_docs(docs: list):
    # adding random documents improves performance 
    random_docs = [
        "Video game monetization is a type of process that a video game publisher can use to generate revenue from a video game product. The methods of monetization may vary between games, especially when they come from different genres or platforms, but they all serve the same purpose to return money to the game developers, copyright owners, and other stakeholders. As the monetization methods continue to diversify, they also affect the game design in a way that sometimes leads to criticism.",
        "Capitalism is an economic system based on the private ownership of the means of production and their operation for profit. Central characteristics of capitalism include capital accumulation, competitive markets, price systems, private property, recognition of property rights, self-interest, economic freedom, meritocracy, work ethic, consumer sovereignty, profit motive, entrepreneurship, commodification, voluntary exchange, wage labor, production of commodities and services, and focus on economic growth.[6][7][8][9][10][11] In a market economy, decision-making and investments are determined by owners of wealth, property, or ability to maneuver capital or production ability in capital and financial markets—whereas prices and the distribution of goods and services are mainly determined by competition in goods and services markets.[12]",
    ]
    docs_as_strings = random_docs + list(doc.page_content for doc in docs)
    return "\n\n".join(docs_as_strings)

def send_prompt_experimental(question: str, system_prompt: str):
    template = system_prompt + """Use the context if relevant to help formatting your answer. If the question is irrelevant to Biblical context and Bible translation, you can refuse to answer.
    
    Context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    # Chain
    rag_chain = (
        {"context": retriever | prepend_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    context_docs = retriever.get_relevant_documents(question)
    context = format_docs(context_docs)

    answer = rag_chain.invoke(question)

    # eval_groundedness(question=question, context=context, answer=answer)
    if not eval_statement_of_faith(question, answer):
        answer = ""
        context = ""

    return {
        'response': answer,
        'context': context
    }

def eval_groundedness(question: str, context: str, answer: str):
    # response = send_prompt_experimental(question, system_prompt)
    
    evaluation_prompt = """
    Evaluate the response below. If the answer is based on the context, respond True. Otherwise, respond False:
    
    Question: {0}

    Context: {1}

    Answer: {2}
    """.format(question, context, answer)
    
    messages = [
        ("user", evaluation_prompt),
    ]
    
    passed = llm.invoke(messages).content
    print(passed)
    
def eval_statement_of_faith(question: str, answer: str):
    # response = send_prompt_experimental(question, system_prompt)
    
    evaluation_prompt = """
    Evaluate the answer below. If the answer contradicts with the following statement of faith, respond False. Otherwise, respond True:
    
    Statement of Faith:
    
    We consider Essential beliefs to be those that define us as believers in Jesus Christ. These cannot be disregarded or compromised.
    The Bible is divinely inspired by God and has final authority.
    God is one and exists in three persons: God the Father, God the Son and God the Holy Spirit.
    Because of the fall of man, all humans are sinful, and in need of salvation.
    The death of Christ is a substitute for sinners and provides for the cleansing of those who believe.
    By God's grace, through faith, man receives salvation as a free gift because of Jesus' death and resurrection.
    The resurrection of all—the saved to eternal life and the lost to eternal punishment.

    Question: {0}

    Answer: {1}
    """.format(question, answer)
    
    messages = [
        ("user", evaluation_prompt),
    ]
    
    passed = llm.invoke(messages).content
    print(passed)

    return passed == "True"
    

    