import os

import streamlit as st

st.title("Invoices Chat")

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from qdrant_client.models import Filter, FieldCondition, DatetimeRange

from dotenv import load_dotenv
load_dotenv()

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "invoices"


@st.cache_resource
def load_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
    )


llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="openai/gpt-oss-120b",
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

chain = prompt | llm | StrOutputParser()


def build_date_filter(date_min, date_max):
    """Filtre Qdrant sur metadata.date :
    - date_min seule  -> factures à partir de cette date
    - date_max seule  -> factures jusqu'à cette date
    - les deux        -> factures entre les deux dates
    - aucune          -> pas de filtre
    """
    if date_min is None and date_max is None:
        return None

    return Filter(must=[
        FieldCondition(
            key="metadata.date",
            range=DatetimeRange(
                gte=date_min.isoformat() if date_min else None,
                lte=date_max.isoformat() if date_max else None,
            ),
        )
    ])



st.sidebar.header("Filtrer par date")
date_min = st.sidebar.date_input("Date min", value=None, format="YYYY-MM-DD")
date_max = st.sidebar.date_input("Date max", value=None, format="YYYY-MM-DD")

question = st.text_input("Pose ta question sur les factures")

if st.button("Envoyer") and question:
    if date_min and date_max and date_min > date_max:
        st.error("La date min doit être avant la date max.")
        st.stop()

    vector_store = load_vector_store()

    documents = vector_store.similarity_search(
        question,
        k=5,
        filter=build_date_filter(date_min, date_max),
    )

    if not documents:
        st.warning("Aucune facture dans cette période.")
    else:
        context = "\n\n".join(doc.page_content for doc in documents)
        answer = chain.invoke({"question": question, "context": context})

        st.subheader("Réponse")
        st.write(answer)

        st.subheader("Factures utilisées")
        for doc in documents:
            m = doc.metadata
            st.write(f"- **{m.get('date')}** | {m.get('bill_to')} | total : {m.get('total')} $ | {m.get('ship_to')}")