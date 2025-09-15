import re
import sys
from pathlib import Path
from typing import Optional


_START_FENCE_RE = re.compile(r"\A\s*```(?:python|py|py3)?\s*\n", re.IGNORECASE)
_END_FENCE_RE = re.compile(r"\n?```\s*\Z")


def _strip_leading_code_fence(code: str) -> str:
    return _START_FENCE_RE.sub("", code, count=1)


def _strip_trailing_code_fence(code: str) -> str:
    return _END_FENCE_RE.sub("", code, count=1)


def _strip_lonely_trailing_triple_quotes(code: str) -> str:
    # Remove a trailing line that is only ''' or """
    lines = code.rstrip().splitlines()
    if not lines:
        return code.strip()
    last = lines[-1].strip()
    if last in {"'''", '"""'}:
        lines = lines[:-1]
        return ("\n".join(lines)).rstrip() + "\n"
    return code


def clean_generated_code(code: Optional[str]) -> str:
    """Clean LLM-generated Python code.

    - Removes leading Markdown code fences like ```python
    - Removes trailing ``` fences
    - Removes stray trailing triple quotes (three single quotes or three double quotes) on the last line
    - Trims leading/trailing whitespace and ensures a single trailing newline
    """
    if not code:
        return ""

    cleaned = code.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    cleaned = _strip_leading_code_fence(cleaned)
    cleaned = _strip_trailing_code_fence(cleaned)
    cleaned = _strip_lonely_trailing_triple_quotes(cleaned)

    # If the entire block is still fenced (rare), unwrap the first fenced block
    if cleaned.strip().startswith("```"):
        parts = re.split(r"\s*```(?:python|py|py3)?\s*\n|\n```\s*", cleaned, maxsplit=2, flags=re.IGNORECASE)
        # parts may be [prefix, content, suffix]; pick the middle if present
        if len(parts) >= 2 and parts[1].strip():
            cleaned = parts[1]

    cleaned = cleaned.strip() + "\n"
    return cleaned


def clean_file(path: Path) -> bool:
    p = Path(path)
    if not p.is_file():
        return False
    original = p.read_text(encoding="utf-8", errors="ignore")
    cleaned = clean_generated_code(original)
    if cleaned != original:
        p.write_text(cleaned, encoding="utf-8")
        return True
    return False


def clean_directory(directory: Path) -> int:
    d = Path(directory)
    count = 0
    for py in d.rglob("*.py"):
        if clean_file(py):
            count += 1
    return count


if __name__ == "__main__":
    # Simple CLI: pass a file or directory path to clean in-place
    if len(sys.argv) < 2:
        print("Usage: python -m utils.code_cleaner <file_or_directory>")
        sys.exit(1)
    target = Path(sys.argv[1])
    if target.is_dir():
        n = clean_directory(target)
        print(f"Cleaned {n} Python files in {target}")
    else:
        ok = clean_file(target)
        print(f"Cleaned: {ok} — {target}")
