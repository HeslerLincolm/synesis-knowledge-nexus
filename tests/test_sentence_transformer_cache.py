"""
Valida la LÓGICA de caché de src.rag.sentence_transformer_embeddings (hash
de texto, reuso entre corridas, recálculo solo de lo que cambió) usando un
modelo simulado -- no requiere que sentence-transformers/torch estén
instalados para correr este test. La carga real del modelo
(paraphrase-multilingual-MiniLM-L12-v2) se prueba aparte, en el entorno
real del usuario, donde sí están las dependencias pesadas instaladas.
"""

import sys
import os
import tempfile
import shutil
from unittest.mock import patch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rag import sentence_transformer_embeddings as ste


class _FakeModel:
    """Simula SentenceTransformer.encode: determinista, basado en hash del
    texto, para poder verificar exactamente qué se recalculó y qué no."""
    def __init__(self):
        self.encode_call_count = 0

    def encode(self, texts, show_progress_bar=False, normalize_embeddings=True):
        self.encode_call_count += 1
        return np.array([[hash(t) % 1000 / 1000.0, 0.5, 0.1] for t in texts])


def test_cache_reused_when_text_unchanged():
    tmpdir = tempfile.mkdtemp()
    try:
        cache_path = os.path.join(tmpdir, "cache.pkl")
        fake_model = _FakeModel()

        with patch.object(ste, "_get_model", return_value=fake_model):
            texts = {"P1": "Texto sobre matemática", "P2": "Texto sobre nefrología"}
            emb1 = ste.build_document_embeddings(texts, cache_path=cache_path)
            assert fake_model.encode_call_count == 1  # primera vez: calcula ambos

            # segunda llamada, mismo texto -> no debe recalcular nada
            emb2 = ste.build_document_embeddings(texts, cache_path=cache_path)
            assert fake_model.encode_call_count == 1  # sigue en 1, se usó la caché
            assert np.array_equal(emb1["P1"], emb2["P1"])
    finally:
        shutil.rmtree(tmpdir)


def test_cache_recomputes_only_changed_document():
    tmpdir = tempfile.mkdtemp()
    try:
        cache_path = os.path.join(tmpdir, "cache.pkl")
        fake_model = _FakeModel()

        with patch.object(ste, "_get_model", return_value=fake_model):
            texts_v1 = {"P1": "Texto original", "P2": "Texto sin cambios"}
            ste.build_document_embeddings(texts_v1, cache_path=cache_path)
            assert fake_model.encode_call_count == 1

            # P1 cambia de texto, P2 sigue igual -> solo debe recalcular P1
            texts_v2 = {"P1": "Texto MODIFICADO", "P2": "Texto sin cambios"}
            ste.build_document_embeddings(texts_v2, cache_path=cache_path)
            assert fake_model.encode_call_count == 2  # una llamada más, no dos
    finally:
        shutil.rmtree(tmpdir)


if __name__ == "__main__":
    test_cache_reused_when_text_unchanged()
    test_cache_recomputes_only_changed_document()
    print("Todos los tests de caché de embeddings pasaron.")
