from dotenv import load_dotenv
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

EMBED_DIM = 384

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

embeddings = HuggingFaceEmbeddings(model_name = EMBEDDING_MODEL)

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150, separators= [
    "\n\n",
    "\n",
    ".",
    " ",
    ""
])


def load_and_chunk_pdf(path:str):
    loader = PyPDFLoader(path)
    pages = loader.load()

    if not pages:
        raise ValueError("No pages could be extracted from the pdf")
    
    chunks = splitter.split_documents(pages)

    chunks = [
        chunk
        for chunk in chunks
        if chunk.page_content and chunk.page_content.strip()
    ]

    if not chunks:
        raise ValueError(
            "No text could be extracted"
            "the pdf maybe scanned/image only"
        )
    return chunks


