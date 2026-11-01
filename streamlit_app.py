from pathlib import Path

import streamlit as st

from rag import ingest_pdf, ask_question


#set the configuration
st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📄",
    layout="centered",
)

# helper function
def save_uploaded_pdf(uploaded_file) -> Path:
    """
    Save the uploaded PDF inside the local uploads folder.
    """
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)

    file_path = uploads_dir / uploaded_file.name
    file_path.write_bytes(uploaded_file.getbuffer())

    return file_path



st.title("📄 PDF RAG Assistant")

st.write(
    "Upload a PDF, ingest it into the vector database, "
    "and ask questions about its contents."
)




st.header("1. Upload a PDF")

uploaded_file = st.file_uploader(
    "Choose a PDF",
    type=["pdf"],
    accept_multiple_files=False,
)


if uploaded_file is not None:

    st.info(f"Selected file: **{uploaded_file.name}**")

    if st.button(
        "📥 Ingest PDF",
        type="primary",
        use_container_width=True,
    ):

        try:
            with st.spinner(
                "Reading PDF, creating chunks, "
                "generating embeddings, and storing them..."
            ):

                # Save uploaded PDF
                pdf_path = save_uploaded_pdf(uploaded_file)

                # Send PDF to our RAG pipeline
                result = ingest_pdf(
                    pdf_path=str(pdf_path),
                    source_id=uploaded_file.name,
                )

            st.success(
                f"Successfully ingested "
                f"{result['ingested']} chunks "
                f"from **{uploaded_file.name}**."
            )

        except Exception as e:

            import traceback

            st.error(
                "Ingestion failed:\n\n"
                + traceback.format_exc()
            )


st.divider()




st.header("2. Ask a question")

question = st.text_input(
    "Your question",
    placeholder="What is this document about?",
)


# Number of chunks to retrieve
top_k = st.slider(
    "Number of chunks to retrieve",
    min_value=1,
    max_value=10,
    value=5,
    step=1,
)



if st.button(
    "🔍 Ask",
    type="primary",
    use_container_width=True,
):

   
    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        try:

            with st.spinner(
                "Searching the documents and generating an answer..."
            ):

                result = ask_question(
                    question=question,
                    top_k=top_k,
                )


            

            st.subheader("Answer")

            st.write(
                result["answer"]
            )


            

            if result["sources"]:

                st.subheader("Sources")

                # Remove duplicate source names
                unique_sources = list(
                    dict.fromkeys(
                        result["sources"]
                    )
                )

                for source in unique_sources:

                    st.write(
                        f"📄 {source}"
                    )


            
            #retrieving metadata
            if result["metadata"]:

                with st.expander(
                    f"Retrieved context "
                    f"({result['num_contexts']} chunks)"
                ):

                    for i, item in enumerate(
                        result["metadata"],
                        start=1,
                    ):

                        source = item.get(
                            "source",
                            "Unknown",
                        )

                        page = item.get(
                            "page",
                            "Unknown",
                        )

                        score = item.get(
                            "score",
                            0,
                        )

                        st.write(
                            f"**Chunk {i}**  \n"
                            f"📄 Source: {source}  \n"
                            f"📖 Page: {page}  \n"
                            f"🔢 Similarity score: {score:.4f}"
                        )

                        st.divider()


        except Exception as e:

            st.error(
                f"Question answering failed:\n\n{e}"
            )




st.divider()

#footer
st.caption(
    "Built with Streamlit + LangChain + "
    "HuggingFace Embeddings + Qdrant + Ollama"
)

