"""Load facts.csv and split it into chunks (free, local; no API calls)."""
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from filing_analyst.config import CHUNK_OVERLAP, CHUNK_SIZE, FACTS_CSV


def load_and_split():
    """Return the chunked LangChain documents built from data/processed/facts.csv."""
    loader = CSVLoader(str(FACTS_CSV))
    data = loader.load()

    rc_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    docs = rc_splitter.split_documents(data)
    return docs
