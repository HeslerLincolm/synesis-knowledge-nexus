# synesis: Academic Knowledge Discovery

Sistema que no solo recupera información académica, sino que **descubre y explica
conexiones** entre piezas de conocimiento institucional, incluyendo conexiones
indirectas que el usuario no buscó explícitamente, cruzando facultades distintas.

---

## Demo rápida

```bash
pip install -r requirements.txt
cp .env.example .env   # Agregar GEMINI_API_KEY para explicaciones reales
streamlit run app.py
```

Consulta de ejemplo: *"identificar antecedentes y capacidades para permanencia estudiantil"*

Sin `GEMINI_API_KEY` configurada, la app corre igual en **modo dry-run**: todo el
pipeline de descubrimiento (RAG, grafo, clustering, scoring, hidden connections)
funciona 100% local y gratis, solo la redacción final de la explicación se
reemplaza por un texto de ejemplo con el mismo formato exacto.

---

## Arquitectura

```
Usuario → Streamlit UI
              ↓
        Query embedding (sentence-transformers, local)
              ↓
   ┌──────────┼──────────────┐
   │          │              │
  RAG    Grafo (NetworkX)  Clustering (K-Means, precalculado)
(Chroma)  jerarquía oficial
   │          │              │
   └────→ Mathematical Scorer ←┘
              ↓
     Cross-Faculty Hidden Connection Detector
     (cruza clusters K-Means × topología del grafo)
              ↓
     Cadena de valor determinista
     NECESIDAD → ANTECEDENTES → PROYECTOS/TESIS →
     INVESTIGADORES/GRUPOS → CAPACIDADES → CURRÍCULO → OPORTUNIDAD
              ↓
     Gemini (ÚNICA llamada por consulta)
     3 bloques exactos: Oportunidad Accionable / Evidencia y
     Naturaleza de la Relación / Ruta de Trazabilidad
```

**Separación en dos fases:**
- `build_index()` — se ejecuta una sola vez al arrancar (carga los 13 CSV, construye
  el grafo, calcula embeddings con caché a disco, corre K-Means y el detector
  cross-faculty sobre *todo* el dataset).
- `answer_query()` — por consulta: retrieval + scoring + selección del hallazgo +
  **una sola llamada a Gemini**. Rápido y barato en cuota.

---

## Dataset

Se trabaja con **Data V1.0 oficial**. De los archivos disponibles, se usan **13**, elegidos para cubrir el flujo completo de descubrimiento sin arrastrar peso innecesario:

| Capa | Archivos usados |
|---|---|
| `01_institution` | faculties, programs, research_groups, research_lines, institutional_capabilities |
| `02_people_curriculum` | researchers, researcher_group |
| `03_knowledge_needs` | institutional_needs, projects, theses, thesis_advisor, publications, publication_researcher |

---

## Decisiones técnicas clave

**Centralidad: degree centrality, no PageRank.** Validado empíricamente sobre un
grafo de prueba antes de tocar datos reales: PageRank no discrimina en grafos
pequeños/poco densos (rango medido: 0.000 en el grafo de validación). Degree
centrality sí discrimina y es más fácil de explicar ("cuántas conexiones directas
tiene").

**Embeddings: sentence-transformers para indexar y consultar, Gemini solo para
explicar.** La similitud coseno solo es válida si consulta y documentos están en el
mismo espacio vectorial — mezclar embeddings de dos modelos distintos habría dado
resultados sin significado real. Se usa `paraphrase-multilingual-MiniLM-L12-v2`
(local, gratis, sin límite de cuota) para todo lo que sea comparación semántica;
Gemini se reserva exclusivamente para la única llamada de explicación final por
consulta.

**Embeddings calculados sobre título + keywords, no sobre la descripción
completa.** Se detectó empíricamente que `projects.csv`/`theses.csv`/
`publications.csv` generan sus descripciones con plantillas de texto repetidas
(~9% de los documentos comparten frases idénticas como *"evidencia trazable y
validación contextual"*), lo que inflaba la similitud semántica por redacción
compartida en vez de por contenido real. La columna `keywords` es la señal de
contenido más limpia disponible.

**Hidden Connection Discovery cruza dos señales, no una.** Se combina el cluster
semántico (K-Means sobre embeddings) con la facultad real de cada nodo (vía el
grafo institucional). Un documento que cae en un cluster dominado por una facultad
distinta a la suya es candidato a conexión oculta — validado end-to-end con
hallazgos reales.

**Ruta de Trazabilidad acotada.**. Se limita a 5 entidades relacionadas por lado,
conservando el conteo total, no se pierde información, solo se deja de listar
todo.

---

## Trazabilidad estricta

Todo nodo del grafo puede resolver su cadena completa de IDs canónicos oficiales
(`researcher_id`, `group_id`, `faculty_id`, `capability_id`, `project_id`, etc.)
vía `get_trace_ids()`, sin importar si la jerarquía viene embebida directo (como en
`projects.csv`, que declara `faculty_id`/`program_id`/`group_id` a la vez) o se
infiere por traversal (como una publicación, cuya facultad se deriva vía
autor → grupo → facultad). Esta traza acompaña cada resultado del Mathematical
Scorer y cada entrada de la Ruta de Trazabilidad final. El sistema nunca devuelve
solo texto.

---

## Limitaciones conocidas

- **La distancia en grafo entre facultades es siempre 4.** La jerarquía
  institucional conecta las facultades entre sí únicamente a través de una única
  Universidad compartida — no hay otro camino corto posible. Esto no afecta la
  etiqueta `EXPLICIT_CONNECTION`/`INFERRED_CONNECTION` (que sigue siendo correcta),
  pero significa que ese número específico aporta poco matiz adicional.
- ** Tres grupos de investigación** (`GRP-022`, `GRP-023`, `GRP-024`) tienen
  `faculty_id` vacío en la fuente original — confirmado como dato real, no un error
  de parseo. El sistema lo reporta como advertencia no bloqueante en vez de fallar.
- **El "paralelismo de 3 búsquedas"** es parcial y así se documenta: RAG y las
  señales de grafo estáticas (centralidad) sí corren en paralelo genuinamente; el
  análisis de proximidad de grafo depende del resultado del RAG (necesita un
  ancla) y por definición no puede ser independiente.
- Los eslabones `CAPACIDADES`/`CURRÍCULO` de la cadena de valor dependen de
  `institutional_capabilities.csv`, que no declara una llave foránea explícita a
  Facultad/Grupo en Data V1.0 — su relevancia se vincula mediante la arista
  `APPLIES_CAPABILITY` que el sistema construye, no mediante un campo del dataset
  original.

---

## Estructura del proyecto

```
synesis/
├── app.py                          # UI Streamlit (5 pestañas del flujo completo)
├── data/
│   ├── official/                   # Data V1.0 oficial, sin modificar
│   └── processed/                  # Chroma, caché de embeddings (regenerable)
├── src/
│   ├── ingestion/official_adapter.py   # Lee los 13 CSV oficiales
│   ├── graph/                          # Construcción y consultas del grafo
│   ├── rag/                             # Vector store + embeddings reales
│   ├── clustering/                      # K-Means
│   ├── scoring/relevance.py             # R(v|q)
│   ├── discovery/                       # Cross-faculty detector + cadena de valor
│   ├── llm/                             # Prompt + cliente Gemini
│   └── orchestrator.py                  # Pipeline completo, build_index + answer_query
├── tests/                           # 9 archivos, suite completa
└── experiments/scoring_poc.py       # Validación aislada de R(v|q), previa a integración
```

## Tests

```bash
python -m pytest tests/ -v
```

Cobertura: dataset, grafo, clustering, contrato funcional (trazabilidad, jerarquía,
cross-faculty, prompt), adaptador de datos oficial, orquestador, UI (headless vía
`streamlit.testing`), y caché de embeddings.
