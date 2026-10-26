"""Headless run of the Streamlit app (single-review path). Skipped without the saved models."""
import os

import pytest

from artifacts import MODEL_PATH

APP = os.path.join(os.path.dirname(__file__), "..", "src", "app.py")


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="model_pipeline.joblib not present")
def test_single_review_prediction_shows_stage_table():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    assert not at.exception
    at.text_area[0].input("I bought this for my son. He loves it, though the battery could be better.")
    at.button[0].click()
    at.run()
    assert not at.exception, at.exception
    assert any("Prediction:" in el.value for el in list(at.success) + list(at.warning))
    stage_table = at.table[0].value
    assert stage_table["Check"].iloc[0].startswith("Stage A")
    assert any("logit" in c.value for c in at.caption)
