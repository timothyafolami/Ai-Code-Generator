import os 
from pathlib import Path
import sys
# Ensure project root is on sys.path so absolute imports work when running as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import pandas as pd
from utils.llm_clients import _base_openai
from utils.aws_bedrock import get_llm
from scripts.markdown_loader import load_markdown_file
from loguru import logger
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from analysis.ai_planner import generate_analysis_plans
from utils.code_cleaner import clean_generated_code


# Build an absolute path to the code generator prompt (fix extension)
prompt_path = ROOT / "prompts" / "code_generator_prompt.md"
code_generator_prompt = load_markdown_file(prompt_path)
llm = get_llm()

# Create the code generator prompt template
code_generator_chain = PromptTemplate(
    template="""
    {master_prompt}

    ## Analysis Plan
    {analysis_plan}



    ## Instructions
    Write Python code using pandas to perform the analysis task described in the Analysis Plan section above.
    The code should be self-contained and executable, assuming the DataFrame is named `df`.
    Use comments to explain each step of the code.

    Return only the Python code, without any additional text or explanation.
    """,
        input_variables=["analysis_plan"],  # Dynamic inputs
        partial_variables={
            "master_prompt": code_generator_prompt  # Static side input
        }
    ) | llm | StrOutputParser()



# Create the code generator class
class CodeGenerator:
    def __init__(self, llm=None):
        """Initialize the code generator with an LLM."""
        self.llm = llm
        self.chain = code_generator_chain
    
    def generate_code(self, df: pd.DataFrame, analysis_plan: str) -> str:
        """Generate Python code based on the analysis plan."""
        try:
            # Step 1: Generate code using the provided analysis plan
            code = self.chain.invoke({"analysis_plan": analysis_plan})
            logger.debug("Generated Python code for analysis plan.")
            return code
        except Exception as e:
            logger.error(f"Error generating code: {e}")
  
  
            raise

if __name__ == "__main__":
    # Example usage
    data_path = "data/synthetic_business_data.csv"
    df = pd.read_csv(data_path)
    code_directory = ROOT / "generated_code"
    os.makedirs(code_directory, exist_ok=True)

    # Step 1: Generate analysis plans
    plans = generate_analysis_plans(df=df, llm=llm)
    for i, plan in enumerate(plans, 1):
        print(f"Analysis Plan {i}: {plan}")

        # Step 2: Generate code for each plan
        code_gen = CodeGenerator(llm=llm)
        # Ensure plan is a JSON string for templating
        plan_str = plan if isinstance(plan, str) else json.dumps(plan, ensure_ascii=False, indent=2)
        code = code_gen.generate_code(df, plan_str)
        code = clean_generated_code(code)
        print(f"Generated Code for Plan {i}:\n{code}\n")
        # Step 3: Save the generated code to a .py file
        code_file = code_directory / f"analysis_plan_{i}.py"
        with open(code_file, "w") as f:
            f.write(code)
        logger.info(f"Saved generated code to {code_file}")
