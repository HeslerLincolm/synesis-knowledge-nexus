"""
Cliente de Gemini para la ÚNICA llamada final del pipeline (explicación).

dry_run=True (default en desarrollo): no llama a la red, solo valida que el
prompt esté bien formado y devuelve un texto de ejemplo -- útil para probar
todo el pipeline sin gastar cuota mientras no haya GEMINI_API_KEY configurada.

dry_run=False: llama de verdad a la API vía el SDK google-genai. Requiere
GEMINI_API_KEY en el entorno (ver src/config.validate_config).
"""

from __future__ import annotations

from src.config import GEMINI_API_KEY, GEMINI_MODEL, validate_config
from src.llm.prompts import validate_output_structure


class GeminiCallError(RuntimeError):
    pass


def generate_explanation(prompt: str, dry_run: bool = True) -> str:
    if dry_run:
        return _dry_run_response()

    validate_config()  # falla rápido y claro si falta GEMINI_API_KEY

    try:
        from google import genai
    except ImportError as e:
        raise GeminiCallError(
            "El paquete 'google-genai' no está instalado. Ejecuta: pip install google-genai"
        ) from e

    client = genai.Client(api_key=GEMINI_API_KEY)
    try:
        response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    except Exception as e:
        # Manejo explícito de rate limits / errores de cuota, como pide el
        # prompt original (minimizar llamadas, manejar errores de rate limit).
        raise GeminiCallError(f"Error llamando a Gemini: {e}") from e

    text = response.text
    validation = validate_output_structure(text)
    if not validation["all_sections_present"]:
        raise GeminiCallError(
            f"La respuesta de Gemini no siguió el formato exigido. "
            f"Faltan secciones: {validation['missing_sections']}"
        )
    return text


def _dry_run_response() -> str:
    """Respuesta de ejemplo, con la estructura exacta exigida, para probar
    el pipeline completo (UI incluida) sin llamar a la API real."""
    return """## Oportunidad Accionable
[DRY RUN -- sin llamada real a Gemini] El Grupo de IA aplicada a Nefrología (Facultad de \
Ingeniería) investiga deep learning para diagnóstico renal, y su trabajo resulta \
semánticamente cercano al del Grupo de Teoría de Aproximación (Facultad de Ciencias), \
sin que exista ninguna colaboración explícita entre ambos grupos hasta ahora.

## Evidencia y Naturaleza de la Relación
Esta es una INFERRED_CONNECTION: no existe arista en el grafo institucional entre ambos \
grupos, la similitud proviene únicamente del espacio semántico (embeddings) y de que \
ambos documentos cayeron en el mismo cluster de K-Means. Se presenta como hipótesis a \
validar, no como hecho comprobado.

## Ruta de Trazabilidad
- paper_id: P4 (Facultad de Ingeniería, F2)
- paper_id: P19 (Facultad de Ciencias, F1)
- researcher_id: R3, R1
- group_id: G2, G1
- capability_id: C2, C1
"""


if __name__ == "__main__":
    from src.ingestion.loader import load_dataset
    from src.graph.builder import build_graph
    from src.rag.synthetic_embeddings import build_synthetic_embeddings
    from src.clustering.kmeans_communities import run_kmeans
    from src.discovery.cross_faculty import detect_cross_faculty_connections
    from src.discovery.value_chain import build_value_chain
    from src.llm.prompts import build_explanation_prompt

    dataset = load_dataset()
    graph = build_graph(dataset)
    embeddings = build_synthetic_embeddings(dataset)

    labels = {}
    for p in dataset.papers.values():
        labels[p.id] = p.title
    for t in dataset.theses.values():
        labels[t.id] = t.title
    for pr in dataset.projects.values():
        labels[pr.id] = pr.name

    document_ids = list(dataset.papers.keys()) + list(dataset.theses.keys()) + list(dataset.projects.keys())
    document_embeddings = {i: embeddings[i] for i in document_ids}
    assignments = run_kmeans(document_embeddings, n_clusters=5)
    findings = detect_cross_faculty_connections(graph, assignments, embeddings, labels)

    p4_finding = next(f for f in findings if f["outlier_id"] == "P4")
    query = "I want to research deep learning applied to kidney disease."
    chain = build_value_chain(graph, dataset, query, p4_finding)
    prompt = build_explanation_prompt(query, chain)

    response = generate_explanation(prompt, dry_run=True)
    print(response)

    validation = validate_output_structure(response)
    print("\n--- Validación de estructura ---")
    print(validation)
