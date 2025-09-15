import ast
import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from loguru import logger

from utils.code_cleaner import clean_generated_code
from utils.code_executor import DataAnalysisCodeExecutor


def find_analysis_function_name(code: str) -> Optional[str]:
    """Heuristically find the analysis function name in generated code.

    Preference order:
    1) analyze_data(df)
    2) First function whose first parameter is named 'df'
    3) First function defined
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    candidate_first_param_df: Optional[str] = None
    first_defined: Optional[str] = None

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            name = node.name
            if first_defined is None:
                first_defined = name
            if name == "analyze_data":
                return name
            if node.args.args:
                if node.args.args[0].arg == "df" and candidate_first_param_df is None:
                    candidate_first_param_df = name

    return candidate_first_param_df or first_defined


def to_json_safe(obj: Any) -> Any:
    """Convert common analysis outputs to JSON-serializable structures."""
    try:
        import numpy as np  # type: ignore
    except Exception:
        np = None  # type: ignore

    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.to_list()
    if isinstance(obj, pd.Index):
        return list(obj)

    if np is not None:
        if isinstance(obj, np.ndarray):  # type: ignore[attr-defined]
            return obj.tolist()
        if isinstance(obj, (np.integer,)):  # type: ignore[attr-defined]
            return int(obj)
        if isinstance(obj, (np.floating,)):  # type: ignore[attr-defined]
            return float(obj)
        if isinstance(obj, (np.bool_,)):  # type: ignore[attr-defined]
            return bool(obj)

    for attr in ("isoformat",):
        if hasattr(obj, attr):
            try:
                return getattr(obj, attr)()
            except Exception:
                pass

    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [to_json_safe(v) for v in obj]

    return str(obj)


def run_generated_directory(
    data_path: Path,
    code_dir: Path,
    out_dir: Path,
    timeout: int = 120,
) -> Dict[str, Any]:
    """Execute each generated analysis file using the executor and save JSON results.

    Returns a summary dictionary with per-file status and output paths.
    """
    code_dir = Path(code_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    logger.info(f"Loaded data: {data_path} shape={df.shape}")

    exec_summary: Dict[str, Any] = {"files": []}
    executor = DataAnalysisCodeExecutor(timeout=timeout)

    for py_file in sorted(code_dir.glob("*.py")):
        logger.info(f"Executing {py_file.name}")
        raw_code = py_file.read_text(encoding="utf-8", errors="ignore")
        code = clean_generated_code(raw_code)

        func_name = find_analysis_function_name(code)
        if not func_name:
            err = f"No function found in {py_file.name}"
            logger.error(err)
            exec_summary["files"].append({"file": py_file.name, "success": False, "error": err})
            continue

        harness = f"\n\nanalysis_result = {func_name}(df)\n"
        full_code = code + harness

        res = executor.execute(full_code, context={"df": df})

        result_payload: Dict[str, Any] = {
            "file": py_file.name,
            "function": func_name,
            "success": res.get("success", False),
            "execution_time": res.get("execution_time", 0),
            "output": res.get("output", "")[:5000],
        }

        if res.get("success"):
            analysis_result = res.get("locals", {}).get("analysis_result")
            result_payload["result"] = to_json_safe(analysis_result)
            out_path = out_dir / (py_file.stem + ".json")
            out_path.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")
            result_payload["saved_to"] = str(out_path)
            logger.info(f"Saved results to {out_path}")
        else:
            result_payload["error"] = res.get("error")
            result_payload["traceback"] = res.get("traceback")
            logger.error(f"Execution failed for {py_file.name}: {res.get('error')}")

        exec_summary["files"].append(result_payload)

    summary_path = out_dir / "run_summary.json"
    summary_path.write_text(json.dumps(exec_summary, indent=2), encoding="utf-8")
    logger.info(f"Run summary saved to {summary_path}")

    return exec_summary

