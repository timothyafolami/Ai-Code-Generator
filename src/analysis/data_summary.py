import pandas as pd
import json
from io import StringIO
from typing import Any, Dict


def generate_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a JSON-serializable summary of the DataFrame.
    """
    # Capture df.info() into a string
    info_buf = StringIO()
    df.info(buf=info_buf)
    data_info = info_buf.getvalue()

    # Convert non-JSON-serializable objects to plain Python types
    desc_df = df.describe(include="all")
    # Replace NaN/NaT with None for strict JSON compatibility
    desc_clean = desc_df.astype(object).where(pd.notnull(desc_df), None)
    data_stats = desc_clean.to_dict()
    data_columns = df.columns.tolist()
    data_missing = df.isnull().sum().to_dict()
    data_duplicates = int(df.duplicated().sum())
    rows, cols = df.shape
    data_shape = [int(rows), int(cols)]

    # the summary in one piece
    data_summary: Dict[str, Any] = {
        "info": data_info,
        "stats": data_stats,
        "columns": data_columns,
        "missing": data_missing,
        "duplicates": data_duplicates,
        "shape": data_shape,
    }
    return data_summary


if __name__ == "__main__":
    # Example usage
    data_path = "data/synthetic_business_data.csv"
    df = pd.read_csv(data_path)

    summary = generate_data_summary(df)
    print(json.dumps(summary, indent=2))
    
