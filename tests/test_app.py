"""
Smoke test de la UI: la app debe cargar sin excepciones y debe poder
procesar la consulta de demo del prompt original sin errores, dry_run
por defecto (no requiere GEMINI_API_KEY).
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from streamlit.testing.v1 import AppTest


def test_app_loads_without_exceptions():
    at = AppTest.from_file(
        os.path.join(os.path.dirname(__file__), "..", "app.py"), default_timeout=120
    )
    at.run()
    assert not at.exception


def test_demo_query_runs_without_exceptions():
    at = AppTest.from_file(
        os.path.join(os.path.dirname(__file__), "..", "app.py"), default_timeout=120
    )
    at.run()
    at.text_input[0].set_value("identificar antecedentes y capacidades para permanencia estudiantil")
    at.button[0].click()
    at.run()
    assert not at.exception
    assert len(at.tabs) == 5


if __name__ == "__main__":
    test_app_loads_without_exceptions()
    test_demo_query_runs_without_exceptions()
    print("Todos los tests de la UI pasaron.")
