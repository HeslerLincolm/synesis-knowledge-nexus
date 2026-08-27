"""
Búsqueda semántica sobre la colección de ChromaDB. Responde: "¿qué
información relevante existe?" (RAG puro, sin grafo ni clustering
todavía -- eso lo cruza el orquestador después).

Cada resultado ya trae su metadata con los IDs canónicos (insertados por
src.rag.vector_store), así que no hace falta volver a tocar el grafo para
saber a qué facultad/grupo pertenece cada documento recuperado.
"""

from __future__ import annotations
import numpy as np
import chromadb


def retrieve(
    collection: chromadb.Collection,
    query_embedding: np.ndarray,
    top_k: int = 5,
) -> list[dict]:
    result = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for i in range(len(result["ids"][0])):
        distance = result["distances"][0][i]
        hits.append({
            "id": result["ids"][0][i],
            "text": result["documents"][0][i],
            "trace_ids": result["metadatas"][0][i],
            # Chroma con espacio "cosine" devuelve distancia = 1 - similitud
            "semantic_similarity": round(1 - distance, 3),
        })
    return hits


if __name__ == "__main__":
    from src.ingestion.loader import load_dataset
    from src.graph.builder import build_graph
    from src.rag.synthetic_embeddings import build_synthetic_embeddings
    from src.rag.vector_store import get_client, build_collection

    dataset = load_dataset()
    graph = build_graph(dataset)
    embeddings = build_synthetic_embeddings(dataset)

    texts = {}
    for p in dataset.papers.values():
        texts[p.id] = f"{p.title}\n{p.abstract}"
    for t in dataset.theses.values():
        texts[t.id] = f"{t.title}\n{t.abstract}"
    for pr in dataset.projects.values():
        texts[pr.id] = f"{pr.name}\n{pr.description}"

    client = get_client(persist_dir="data/processed/chroma_db_test")
    collection = build_collection(client, graph, texts, embeddings)

    # Query de demo: reusamos el embedding de P4 como proxy de la consulta
    # "deep learning applied to kidney disease" (con Gemini real, aquí iría
    # el embedding calculado de ese texto).
    query_embedding = embeddings["P4"]
    hits = retrieve(collection, query_embedding, top_k=5)

    print("Top-5 resultados para la consulta de demo:")
    for h in hits:
        print(f"\n[{h['id']}] sim={h['semantic_similarity']}")
        print(f"  faculty_id={h['trace_ids'].get('faculty_id')}  group_id={h['trace_ids'].get('group_id')}")
