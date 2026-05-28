# tests/test_llm.py
import os
from src.utils.llm import create_llm


def test_create_llm_default_model():
    llm = create_llm()
    assert llm is not None
    assert llm.model_name == "deepseek-chat"


def test_create_llm_custom_model():
    llm = create_llm(model="deepseek-chat")
    assert llm.model_name == "deepseek-chat"


def test_create_llm_with_base_url():
    llm = create_llm(base_url="https://api.deepseek.com/v1")
    assert llm is not None
