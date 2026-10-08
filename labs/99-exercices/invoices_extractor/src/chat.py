import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

from dotenv import load_dotenv
load_dotenv()

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "invoices"

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="openai/gpt-oss-20b",
)

prompt = ChatPromptTemplate.from_template(
    '''
    You are a helpful assistant
    you always answer the question given the invoices in the context
    if you don't have an answer just say i don't know

    question :
    {question}

    context:
    {context}
    '''
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    url=QDRANT_URL,
    collection_name=COLLECTION_NAME,
)

chain = prompt | llm | StrOutputParser()


def build_filter(bill_to, date, min_total, max_total):
    conditions = []

    if bill_to:
        conditions.append(FieldCondition(key="metadata.bill_to", match=MatchValue(value=bill_to)))

    if date:
        conditions.append(FieldCondition(key="metadata.date", match=MatchValue(value=date)))

    if min_total or max_total:
        conditions.append(FieldCondition(
            key="metadata.total",
            range=Range(
                gte=float(min_total) if min_total else None,
                lte=float(max_total) if max_total else None,
            ),
        ))

    return Filter(must=conditions) if conditions else None


def format_docs(documents):
    return "\n\n".join(doc.page_content for doc in documents)


while True:
    question = input('\nAsk something : ')
    if question.lower() in ['/quit', '/bye']:
        print("Exiting the program.")
        break

    bill_to = input('Filter bill to : ')
    date = input('Filter date (YYYY-MM-DD) : ')
    min_total = input('Filter total min : ')
    max_total = input('Filter total max : ')

    documents = vector_store.similarity_search(
        question,
        k=5,
        filter=build_filter(bill_to, date, min_total, max_total),
    )

    answer = chain.invoke({
        'question': question,
        'context': format_docs(documents),
    })
    print('AI : ', answer)