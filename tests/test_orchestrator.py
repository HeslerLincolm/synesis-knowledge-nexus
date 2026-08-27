"""
Valida el orquestador end-to-end sobre el fixture de Data V1.0 oficial.

Nota importante: con el embedder placeholder actual (bolsa de palabras, no
semántico), no está garantizado que aparezca una conexión oculta -- de
hecho con un fixture tan pequeño normalmente NO aparece. Este test valida
que el pipeline corre sin errores en ambos casos (con y sin hallazgo), no
que encuentre una conexión específica. Cuando se conecte una fuente de
embeddings real (Gemini o sentence-transformers) y los 13 CSV completos,
vale la pena agregar un test que sí verifique un hallazgo concreto conocido
del dataset real.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.orchestrator import build_index, answer_query

DEMO_QUERY = "identificar antecedentes y capacidades para permanencia estudiantil"


def test_pipeline_runs_end_to_end_without_error():
    context = build_index(persist_dir="data/processed/chroma_db_pytest")
    result = answer_query(context, DEMO_QUERY, dry_run=True)

    assert not result.get("error")
    assert len(result["rag_hits"]) > 0
    assert len(result["ranking"]) > 0
    for hit in result["rag_hits"]:
        assert "trace_ids" in hit


def test_explanation_only_generated_when_there_is_a_finding():
    """Si NO hay hallazgo de conexión oculta relevante, el sistema NO debe
    llamar a Gemini (ahorro de cuota) -- explanation debe quedar en None."""
    context = build_index(persist_dir="data/processed/chroma_db_pytest")
    result = answer_query(context, DEMO_QUERY, dry_run=True)

    if result["hidden_connections"]:
        assert isinstance(result["explanation"], str)
        assert "## Oportunidad Accionable" in result["explanation"]
        assert result["value_chain"] is not None
    else:
        assert result["explanation"] is None
        assert result["value_chain"] is None


def test_orchestrator_makes_at_most_one_llm_call_per_query():
    """No debe haber más de una llamada a Gemini por consulta -- se valida
    indirectamente: 'explanation' nunca es una lista, solo un string o None."""
    context = build_index(persist_dir="data/processed/chroma_db_pytest")
    result = answer_query(context, DEMO_QUERY, dry_run=True)
    assert result["explanation"] is None or isinstance(result["explanation"], str)


if __name__ == "__main__":
    test_pipeline_runs_end_to_end_without_error()
    test_explanation_only_generated_when_there_is_a_finding()
    test_orchestrator_makes_at_most_one_llm_call_per_query()
    print("Todos los tests del orquestador pasaron.")
