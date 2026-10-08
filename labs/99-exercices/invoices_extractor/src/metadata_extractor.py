import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from pydantic import BaseModel, Field

from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="openai/gpt-oss-120b",
)


class InvoiceMetadata(BaseModel):
    date: str = Field(description="Invoice date in format YYYY-MM-DD")
    bill_to: str = Field(description="Name of the customer in 'Bill To'")
    ship_to: str = Field(description="Full address in 'Ship To'")
    total: float = Field(description="Total amount, number only")
    shipping: float = Field(description="Shipping cost, number only")
    discount: float = Field(description="Discount amount, number only, 0 if there is no discount")
    amount: float = Field(description="Balance Due, number only")



metadata_llm = llm.with_structured_output(InvoiceMetadata, method="json_schema")

prompt = ChatPromptTemplate.from_template(
    '''
    You are an invoice analysis assistant.
    Extract the metadata of the following invoice.
    In the text, labels (Date, Total, ...) and their values can be separated:
    match them by their order.

    invoice :
    {text}
    '''
)

chain = (prompt | metadata_llm).with_retry(stop_after_attempt=3)


def extract_invoice_metadata(text: str) -> dict:
    """Texte d'une facture -> dictionnaire de metadata."""
    return chain.invoke({"text": text}).model_dump()


if __name__ == "__main__":
    import sys
    documents = PyPDFLoader(sys.argv[1]).load()
    text = "\n".join(document.page_content for document in documents)
    print(text)
    print("-" * 40)
    print(extract_invoice_metadata(text))