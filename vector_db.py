import uuid 
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue
)

from data_loader import EMBED_DIM, embeddings

class QdrantStorage:
    def __init__(self, url: str="http//localhost:6333",
                 collection:str ="docs",
                 dim:int=EMBED_DIM,
                 ):
        self.client=QdrantClient(
            url=url,
            timeout = 30,
        )
        self.collection = collection
        #if collection does not exists
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=dim,
                    distance=Distance.COSINE,
                ),
            )
        self.store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection,
            embedding=embeddings,
        )    
    
    def delete_source(self, source_id:str) -> None:
        self.client.delete(
            collection_name = self.collection,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="metadata.source_id",
                        match=MatchValue(value=source_id),
                    )
                ]
            )

        )



    def upsert(self, documents:list[Document], source_id:str) -> int:
        if not documents:
            return 0
        self.delete_source(source_id)

        cleaned_documents = []

        for index, doc in enumerate(documents):
            metadata= dict(doc.metadata)

            metadata["source_id"] = source_id
            metadata["chunk_index"] = index

            if "page" in metadata:
                metadata["page"] = int(metadata["page"]) + 1

                cleaned_documents.append(
                    Document(
                        page_content = doc.page_content,
                        metadata=metadata
                    )
                )
        #deterministic ids
        ids = [
            str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{source_id}:{index}",
                )
            )
            for index in range(len(cleaned_documents))
        ]

        self.store.add_documents(
            documents=cleaned_documents,
            ids=ids,
        )

        return len(cleaned_documents)
    
    def search(
            self, 
            query:str,
            top_k:int=5,
    ) -> dict:
        results = self.store.similarity_search_with_score(
            query,
            k=top_k,
        )

        contexts = []
        sources = []
        scores = []
        metadata = []

        for doc, score in results:
            if not doc.page_content.strip():
                continue
            contexts.append(doc.page_content)
            source = doc.metadata.get(
                "source_id",
                "Unknown source",
            )
            sources.append(source)
            scores.append(float(score))

            metadata.append(
                {
                    "source" : source,
                    "page" : doc.metadata.get("page"),
                    "chunk_index" : doc.metadata.get("chunk_index"),
                    "score" : float(score),
                }
            )
            return{
                "contexts" : contexts,
                "sources":sources,
                "scores":scores,
                "metadata":metadata,
            }
    def as_retriever(self, top_k:int=5):
        return self.store.as_retriever(
            search_kwargs={
                "k":top_k,
            }
        )