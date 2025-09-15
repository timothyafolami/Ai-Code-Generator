import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from loguru import logger
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Ensure src on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.llm_clients import _base_openai
from scripts.markdown_loader import load_markdown_file
from utils.code_cleaner import clean_generated_code
from utils.code_executor import DataAnalysisCodeExecutor


prompt_path = ROOT / "prompts" / "code_fixer_prompt.md"
CODE_FIXER_PROMPT = load_markdown_file(prompt_path)


fix_prompt = PromptTemplate(
    template="""
{master_prompt}

# Runtime Error
{error_text}

# Original Code
{original_code}
""",
    input_variables=["error_text", "original_code"],
    partial_variables={"master_prompt": CODE_FIXER_PROMPT},
)


def fix_code_once(original_code: str, error_text: str) -> str:
    llm = _base_openai()
    chain = fix_prompt | llm | StrOutputParser()
    fixed = chain.invoke({
        "error_text": error_text,
        "original_code": original_code,
    })
    return clean_generated_code(fixed)


def execute_code_file(py_path: Path, df: pd.DataFrame) -> Dict[str, Any]:
    code = clean_generated_code(py_path.read_text(encoding="utf-8", errors="ignore"))
    # Append harness to call the analysis function; try common names
    from utils.generated_runner import find_analysis_function_name
    func = find_analysis_function_name(code)
    if not func:
        return {"success": False, "error": "No function found to execute"}
    harness = f"\nanalysis_result = {func}(df)\n"
    exec_code = code + harness
    ex = DataAnalysisCodeExecutor(timeout=120)
    res = ex.execute(exec_code, context={"df": df})
    return res


def fix_and_rerun(
    py_path: Path,
    df: pd.DataFrame,
    first_error: str,
    max_attempts: int = 2,
) -> Dict[str, Any]:
    """Attempt to fix a failing code file up to max_attempts using the LLM."""
    orig = py_path.read_text(encoding="utf-8", errors="ignore")
    last_error = first_error
    for attempt in range(1, max_attempts + 1):
        logger.info(f"Fix attempt {attempt} for {py_path.name}")
        fixed = fix_code_once(orig, last_error)
        py_path.write_text(fixed, encoding="utf-8")
        res = execute_code_file(py_path, df)
        if res.get("success"):
            logger.info(f"Fix successful for {py_path.name} on attempt {attempt}")
            return {"success": True, "attempt": attempt, "result": res}
        last_error = res.get("traceback") or res.get("error", "Unknown error")
    logger.error(f"Fix failed after {max_attempts} attempts for {py_path.name}")
    return {"success": False, "attempt": max_attempts, "error": last_error}


if __name__ == "__main__":
    # Defaults and simple positional args: data_path [code_file]
    data_path = ROOT / ".." / "data" / "synthetic_business_data.csv"
    if len(sys.argv) > 1:
        data_path = Path(sys.argv[1])
    df = pd.read_csv(data_path)

    # Optional: fix one file passed as 2nd arg, else scan generated_code
    if len(sys.argv) > 2:
        file_path = Path(sys.argv[2])
        res = execute_code_file(file_path, df)
        if not res.get("success"):
            err = res.get("traceback") or res.get("error", "Unknown error")
            fix_and_rerun(file_path, df, err)
    else:
        code_dir = ROOT / "generated_code"
        for py in sorted(code_dir.glob("*.py")):
            res = execute_code_file(py, df)
            if res.get("success"):
                logger.info(f"OK: {py.name}")
                continue
            err = res.get("traceback") or res.get("error", "Unknown error")
            fix_and_rerun(py, df, err)

