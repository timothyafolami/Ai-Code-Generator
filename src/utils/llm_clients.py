from __future__ import annotations
import os
from typing import Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

load_dotenv()


def _base_openai(model: Optional[str] = None) -> ChatOpenAI:
    """Return a plain ChatOpenAI model configured from env or args.

    - Uses `OPENAI_API_KEY` from env (managed by the SDK)
    - Model defaults to `OPENAI_MODEL` or `gpt-4o-mini`
    """
    mdl = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return ChatOpenAI(model=mdl)


def get_llm(model: Optional[str] = None, temperature: float = 0.0):
    """Normal LLM: returns text. Chain = ChatOpenAI | StrOutputParser."""
    return _base_openai(model=model) | StrOutputParser()


def get_llm_json(model: Optional[str] = None, temperature: float = 0.0):
    """JSON LLM: parses model output as JSON. Chain = ChatOpenAI | JsonOutputParser."""
    return _base_openai(model=model) | JsonOutputParser()


# Global convenience instances
openai_llm = get_llm()
openai_llm_json = get_llm_json()
