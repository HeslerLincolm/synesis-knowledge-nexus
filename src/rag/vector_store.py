"""
Wrapper de ChromaDB. Punto 1 del contrato: la inserción DEBE guardar los
IDs canónicos oficiales (faculty_id, group_id, researcher_id, capability_id,
project_id, etc.) como metadata de cada documento, no solo el texto y su
embedding -- así el Mathematical Scorer puede recuperar la traza completa
directamente desde el resultado de Chroma, sin tener que volver a consultar
el grafo para cada resultado.

Usa embeddings ya calculados (por ahora los sintéticos de
src.rag.synthetic_embeddings; se reemplaza por Gemini embeddings sin tocar
este módulo) -- Chroma se usa aquí solo como almacén + motor de búsqueda
por similitud, no para calcular embeddings él mismo.
"""

from __future__ import annotations
import numpy as np
import chromadb

from src.config import CHROMA_PERSIST_DIR
from src.graph.queries import get_trace_ids
import networkx as nx


COLLECTION_NAME = "synesis_documents"


def get_client(persist_dir: str = CHROMA_PERSIST_DIR) -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=persist_dir)


def build_collection(
    client: chromadb.ClientAPI,
    graph: nx.Graph,
    document_texts: dict[str, str],
    document_embeddings: dict[str, np.ndarray],
    reset: bool = True,
) -> chromadb.Collection:
    """
    document_texts: {entity_id: texto a indexar (título + abstract/descripción)}
    document_embeddings: {entity_id: vector ya calculado}

    Cada documento se inserta con su embedding YA calculado (Chroma no
    recalcula nada) y con metadata = traza completa de IDs canónicos
    obtenida de src.graph.queries.get_trace_ids, más el tipo de entidad.
    """
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids, embeddings, documents, metadatas = [], [], [], []
    for entity_id, text in document_texts.items():
        if entity_id not in document_embeddings:
            continue
        trace = get_trace_ids(graph, entity_id)
        # Chroma exige metadata con tipos primitivos (str/int/float/bool);
        # la traza ya es dict[str, str], se guarda tal cual.
        metadata = dict(trace)
        metadata["entity_type"] = graph.nodes[entity_id].get("type", "unknown") if entity_id in graph else "unknown"

        ids.append(entity_id)
        embeddings.append(document_embeddings[entity_id].tolist())
        documents.append(text)
        metadatas.append(metadata)

    if ids:
        collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    return collection


if __name__ == "__main__":
    from src.ingestion.loader import load_dataset
    from src.graph.builder import build_graph
    from src.rag.synthetic_embeddings import build_synthetic_embeddings

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

    print(f"Documentos insertados: {collection.count()}")
    sample = collection.get(ids=["P4"], include=["metadatas", "documents"])
    print("\nMetadata guardada para P4 (debe incluir faculty_id, group_id, capability_id, etc.):")
    print(sample["metadatas"][0])
