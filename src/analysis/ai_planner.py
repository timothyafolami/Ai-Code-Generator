import os 
from pathlib import Path
import sys
# Ensure project root is on sys.path so absolute imports work when running as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.llm_clients import _base_openai
from scripts.markdown_loader import load_markdown_file
from loguru import logger
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from analysis.data_summary import generate_data_summary
import json
import pandas as pd

# Build an absolute path to the planner prompt (fix extension)
prompt_path = ROOT / "prompts" / "planner_prompt.md"
planner_prompt = load_markdown_file(prompt_path)

llm = _base_openai()

analysis_planner_prompt = PromptTemplate(
    template="""
    {master_prompt}

    ## Input Data Summary
    {data_summary}

    ## Task
    Based on the data summary above, generate exactly 3 analysis plans as a JSON array.

    Return only the JSON array of 3 analysis plans.
    """,
        input_variables=["data_summary"],  # Dynamic input
        partial_variables={
            "master_prompt": planner_prompt  # Static side input
        }
    )


# Create the analysis planner class
class AnalysisPlanner:
    def __init__(self, llm=None):
        """Initialize the analysis planner with an LLM."""
        self.llm = llm
        self.chain = analysis_planner_prompt | self.llm | JsonOutputParser()
    
    def generate_plans(self, df: pd.DataFrame) -> list:
        """Generate 3 analysis plans based on data summary."""
        try:
            # Step 1: Get data summary
            summary = generate_data_summary(df)
            logger.debug("Generated data summary for analysis planning.")

            # Step 2: Convert summary to formatted string
            summary_str = json.dumps(summary, indent=2)
            logger.debug("Serialized data summary to JSON string.")

            # Step 3: Invoke the chain
            logger.debug("Invoking LLM chain for analysis plans...")
            response = self.chain.invoke({"data_summary": summary_str})
            logger.debug("LLM response received for analysis plans.")

            # Step 4: Normalize response to Python list
            if isinstance(response, (list, dict)):
                plans = response
            elif isinstance(response, str):
                plans = json.loads(response)
            else:
                raise TypeError(f"Unexpected LLM response type: {type(response)}")

            # Validate that we got exactly 3 plans
            if len(plans) != 3:
                logger.warning(f"Expected 3 plans, got {len(plans)}")

            return plans

        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON response: {e}")
            logger.error(f"Raw response: {response}")
            return []
        except Exception as e:
            logger.exception(f"Error generating analysis plans: {e}")
            return []

# Convenience function for direct use
def generate_analysis_plans(df: pd.DataFrame, llm=None) -> list:
    """Generate analysis plans for a DataFrame."""
    planner = AnalysisPlanner(llm=llm)
    return planner.generate_plans(df)


if __name__ == "__main__":
    # Example usage
    data_path = "data/synthetic_business_data.csv"
    df = pd.read_csv(data_path)
    
    logger.info(f"Loaded data from {data_path} with shape {df.shape}")

    plans = generate_analysis_plans(df, llm=llm)
    
    for i, plan in enumerate(plans, 1):
        logger.info(f"Plan {i}: {plan}")
