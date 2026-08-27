"""
Ejecutar con: streamlit run app.py

El índice (dataset, grafo, embeddings, Chroma, clustering, cross-faculty
findings) se construye UNA sola vez por sesión (@st.cache_resource) -- no se
recalcula en cada consulta, solo el retrieval + scoring + la llamada final
a Gemini corren por consulta.
"""

import streamlit as st

from src.orchestrator import build_index, answer_query
from src.config import GEMINI_API_KEY

st.set_page_config(page_title="Academic Knowledge Discovery", layout="wide")


@st.cache_resource(show_spinner="Construyendo el índice (dataset, grafo, embeddings, Chroma, clustering)...")
def get_index():
    return build_index(source="official")


def main():
    st.title("🔎 Synesis — Academic Knowledge Discovery")
    st.caption(
        "No solo encontrar información, sino descubrir y explicar conexiones entre "
        "piezas de conocimiento académico. Trabajando sobre **Data V1.0 oficial**."
    )

    with st.sidebar:
        st.warning(
            "⚠️ Embeddings placeholder (bolsa de palabras, no semánticos) -- "
            "Hidden Connection Discovery no va a encontrar conexiones "
            "significativas todavía. Pendiente conectar Gemini embeddings o "
            "sentence-transformers."
        )

        st.divider()
        st.subheader("Configuración")
        dry_run = st.toggle(
            "Modo dry-run (sin llamar a Gemini)",
            value=(GEMINI_API_KEY is None),
            help="Si no tienes GEMINI_API_KEY configurada, deja esto activado. "
                 "El resto del pipeline (RAG, grafo, clustering, scoring, hidden "
                 "connections) corre igual, 100% local.",
        )
        if not GEMINI_API_KEY and not dry_run:
            st.warning("No hay GEMINI_API_KEY configurada. Se forzará dry-run.")
            dry_run = True

        st.divider()
        st.caption("Ejemplo de consulta de demo:")
        st.code("identificar antecedentes y capacidades para permanencia estudiantil", language=None)

    context = get_index()

    query = st.text_input(
        "Consulta de investigación",
        placeholder="I want to research deep learning applied to kidney disease.",
    )
    run = st.button("Buscar", type="primary")

    if not (run and query):
        return

    with st.spinner("Ejecutando RAG + Grafo + Clustering..."):
        result = answer_query(context, query, dry_run=dry_run)

    if result.get("error"):
        st.error(result["error"])
        return

    tabs = st.tabs([
        "1. Resultados (RAG)", "2. Ranking matemático", "3. Conexiones ocultas",
        "4. Ruta de trazabilidad", "5. Explicación final",
    ])

    with tabs[0]:
        st.subheader("Documentos relevantes recuperados por RAG")
        for h in result["rag_hits"]:
            trace = h["trace_ids"]
            with st.container(border=True):
                st.markdown(f"**{h['id']}** — similitud semántica: `{h['semantic_similarity']}`")
                st.caption(h["text"][:280] + ("..." if len(h["text"]) > 280 else ""))
                cols = st.columns(4)
                cols[0].metric("Facultad", trace.get("faculty_id", "—"))
                cols[1].metric("Grupo", trace.get("group_id", "—"))
                cols[2].metric("Investigador", trace.get("researcher_id", "—"))
                cols[3].metric("Competencia", trace.get("capability_id", "—"))

    with tabs[1]:
        st.subheader("Mathematical Relevance Score — R(v|q)")
        st.latex(r"R(v|q) = 0.5 \cdot \text{SemSim} + 0.3 \cdot \text{GraphProx} + 0.2 \cdot \text{Centrality}")
        rows = []
        for r in result["ranking"]:
            rows.append({
                "Entidad": r["node_id"],
                "Similitud semántica": r["semantic_similarity"],
                "Proximidad en grafo": r["graph_proximity"],
                "Centralidad": r["centrality"],
                "R(v|q) final": r["final_relevance"],
                "Facultad": r["trace_ids"].get("faculty_id", "—"),
            })
        st.dataframe(rows, width="stretch", hide_index=True)

    with tabs[2]:
        st.subheader("Hidden Connection Discovery")
        findings = result["hidden_connections"]
        if not findings:
            st.info("No se detectaron conexiones ocultas relevantes para esta consulta.")
        for f in findings:
            badge = "🟢 EXPLICIT_CONNECTION" if f["label"] == "EXPLICIT_CONNECTION" else "🟠 INFERRED_CONNECTION"
            with st.container(border=True):
                st.markdown(f"### {badge}")
                st.markdown(f"**{f['outlier_label']}**  (Facultad `{f['outlier_faculty_id']}`)")
                st.markdown("⬍ similitud semántica: " + f"`{f['semantic_similarity']}`" +
                             (f" · distancia en grafo: `{f['graph_distance']}`" if f["graph_distance"] is not None else " · sin camino en el grafo"))
                st.markdown(f"**{f['partner_label']}**  (Facultad `{f['dominant_faculty_id']}`)")
                if f["label"] == "INFERRED_CONNECTION":
                    st.caption("⚠️ Conexión inferida por similitud semántica y clustering — no es un hecho comprobado en el grafo institucional.")

    with tabs[3]:
        st.subheader("Ruta de trazabilidad (IDs canónicos)")
        chain = result["value_chain"]
        if not chain:
            st.info("Sin cadena de valor para esta consulta (no hubo hallazgo de conexión oculta).")
        else:
            for step_name in ["NECESIDAD", "ANTECEDENTES", "PROYECTOS_TESIS",
                               "INVESTIGADORES_GRUPOS", "CAPACIDADES", "OPORTUNIDAD"]:
                with st.expander(step_name, expanded=(step_name == "OPORTUNIDAD")):
                    st.json(chain[step_name])

    with tabs[4]:
        st.subheader("Explicación final")
        if result["explanation"]:
            if dry_run:
                st.caption("⚠️ Modo dry-run: esta respuesta es un ejemplo fijo, no vino de Gemini.")
            st.markdown(result["explanation"])
        else:
            st.info("No hubo hallazgo de conexión oculta relevante, así que no se llamó a Gemini "
                    "(el sistema evita llamadas innecesarias a la API).")


if __name__ == "__main__":
    main()
