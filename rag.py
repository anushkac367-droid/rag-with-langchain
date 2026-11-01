import os

from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from data_loader import load_and_chunk_pdf
from vector_db import QdrantStorage


load_dotenv()


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3",
)

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
)


# ---------------------------------------------------------
# LLM
# ---------------------------------------------------------

llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0,
)


# ---------------------------------------------------------
# RAG Prompt
# ---------------------------------------------------------

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a document question-answering assistant.

You must answer the user's question using ONLY the
information contained in the provided context.

Rules:

1. Do not use outside knowledge.
2. Do not make up information.
3. Do not guess.
4. If the context does not contain enough information,
   respond exactly with:

"I don't have enough information from the uploaded documents to answer that."

5. Give a clear and concise answer.
6. When possible, explain the answer using the information
   from the context.

Context:
{context}
""",
        ),
        (
            "human",
            "Question: {question}",
        ),
    ]
)


# ---------------------------------------------------------
# LangChain RAG chain
# ---------------------------------------------------------

rag_chain = RAG_PROMPT | llm | StrOutputParser()



def ingest_pdf(
    pdf_path: str,
    source_id: str,
) -> dict:

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    # Load + split PDF
    documents = load_and_chunk_pdf(pdf_path)

    # Store in Qdrant
    vector_db = QdrantStorage()

    ingested = vector_db.upsert(
        documents,
        source_id,
    )

    return {
        "source": source_id,
        "ingested": ingested,
    }



def ask_question(
    question: str,
    top_k: int = 5,
) -> dict:

    question = question.strip()

    if not question:
        raise ValueError(
            "Question cannot be empty."
        )

    vector_db = QdrantStorage()

    
    search_results = vector_db.search(
        question,
        top_k=top_k,
    )

    contexts = search_results["contexts"]

    sources = search_results["sources"]

    metadata = search_results["metadata"]


    # Nothing found
    if not contexts:

        return {
            "answer": (
                "I don't have enough information from the "
                "uploaded documents to answer that."
            ),
            "sources": [],
            "num_contexts": 0,
            "metadata": [],
        }


    
    context_block = "\n\n".join(
        f"[Document chunk {i + 1}]\n{context}"
        for i, context in enumerate(contexts)
    )


    
    answer = rag_chain.invoke(
        {
            "context": context_block,
            "question": question,
        }
    )


    return {
        "question": question,
        "answer": answer,
        "contexts": contexts,
        "sources": sources,
        "num_contexts": len(contexts),
        "metadata": metadata,
    }