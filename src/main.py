import sys
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from loguru import logger

# Ensure src is on sys.path when running directly
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.llm_clients import _base_openai
from analysis.ai_planner import generate_analysis_plans
from analysis.code_generator import CodeGenerator
from utils.code_cleaner import clean_generated_code
from utils.generated_runner import run_generated_directory
from analysis.code_fixer import fix_and_rerun


def orchestrate_pipeline(data_path: Path) -> Dict[str, Any]:
    """Full pipeline: load -> plan -> codegen -> execute -> save results."""
    data_path = Path(data_path)
    generated_code_dir = ROOT / "generated_code"
    generated_results_dir = ROOT / "generated_results"
    generated_code_dir.mkdir(parents=True, exist_ok=True)
    generated_results_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load data
    df = pd.read_csv(data_path)
    logger.info(f"Loaded data from {data_path} with shape {df.shape}")

    # 2) Plan analyses (LLM JSON)
    planner_llm = _base_openai()
    plans: List[Dict[str, Any]] = generate_analysis_plans(df, llm=planner_llm)
    logger.info(f"Generated {len(plans)} analysis plans")

    # Persist plans for reference
    plans_path = generated_results_dir / "analysis_plans.json"
    plans_path.write_text(json.dumps(plans, indent=2), encoding="utf-8")

    # 3) Generate code for each plan
    code_paths: List[str] = []
    code_gen = CodeGenerator()  # uses configured LLM in module
    for i, plan in enumerate(plans, start=1):
        plan_str = plan if isinstance(plan, str) else json.dumps(plan, ensure_ascii=False, indent=2)
        code = code_gen.generate_code(df, plan_str)
        code = clean_generated_code(code)

        code_file = generated_code_dir / f"analysis_plan_{i}.py"
        code_file.write_text(code, encoding="utf-8")
        code_paths.append(str(code_file))
        logger.info(f"Saved generated code: {code_file}")

    # 4) Execute generated analyses and save JSON results
    run_summary = run_generated_directory(
        data_path=data_path,
        code_dir=generated_code_dir,
        out_dir=generated_results_dir,
        timeout=120,
    )
    
    # 5) Attempt automatic fixes for any failures (up to 2 attempts each), then re-run
    failures = [f for f in run_summary.get("files", []) if not f.get("success")]
    fix_reports: List[Dict[str, Any]] = []
    if failures:
        logger.info(f"Attempting fixes for {len(failures)} failing analyses")
        for f in failures:
            fname = f.get("file")
            if not fname:
                continue
            err_text = f.get("traceback") or f.get("error") or "Unknown error"
            py_path = generated_code_dir / fname
            report = fix_and_rerun(py_path, df, err_text, max_attempts=2)
            report["file"] = fname
            fix_reports.append(report)

        # Re-run directory to write final JSON outputs after fixes
        final_run_summary = run_generated_directory(
            data_path=data_path,
            code_dir=generated_code_dir,
            out_dir=generated_results_dir,
            timeout=120,
        )
    else:
        final_run_summary = run_summary

    return {
        "plans_saved": str(plans_path),
        "code_files": code_paths,
        "results_dir": str(generated_results_dir),
        "initial_run_summary": run_summary,
        "fix_reports": fix_reports,
        "final_run_summary": final_run_summary,
    }


if __name__ == "__main__":
    # Accept data path as first positional arg, else default
    default_data = ROOT.parent / "data" / "synthetic_business_data.csv"
    data_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_data
    logger.info("Starting end-to-end AI analysis pipeline")
    summary = orchestrate_pipeline(data_path)
    logger.info("Pipeline finished")
    print(json.dumps({
        "plans_saved": summary["plans_saved"],
        "code_files": summary["code_files"],
        "results_dir": summary["results_dir"]
    }, indent=2))
