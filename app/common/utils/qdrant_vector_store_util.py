import uuid
from typing import List, Callable, TypeVar, Any
from langchain_huggingface import HuggingFaceEndpointEmbeddings

T = TypeVar("T")
P = TypeVar("P")


class QdrantVectorStoreUtil:

    @staticmethod
    async def batch_upsert(
        embedding_client: HuggingFaceEndpointEmbeddings,
        qdrant_repo: Any,
        entities: List[T],
        text_extractor: Callable[[T], List[str]],
        payload_converter: Callable[[T], P],
        batch_size: int = 10,
    ) -> None:
        await qdrant_repo.ensure_collection()

        points = []
        for entity in entities:
            payload = payload_converter(entity)
            embedding_texts = text_extractor(entity)
            for text in embedding_texts:
                points.append({
                    "id": str(uuid.uuid4()),
                    "embedding_text": text,
                    "payload": payload
                })

        if not points:
            return

        all_texts = [point["embedding_text"] for point in points]
        embeddings = []
        for i in range(0, len(all_texts), batch_size):
            batch_texts = all_texts[i:i + batch_size]
            batch_embeddings = await embedding_client.aembed_documents(batch_texts)
            embeddings.extend(batch_embeddings)

        ids = [point["id"] for point in points]
        payloads = [point["payload"] for point in points]
        await qdrant_repo.upsert_embeddings(ids, embeddings, payloads)