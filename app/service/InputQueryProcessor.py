import os
import sys
import json
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
grandparent_dir = os.path.abspath(os.path.join(parent_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

if grandparent_dir not in sys.path:
    sys.path.append(grandparent_dir)

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from groq import Groq
from app.repository.CompanyRepository import CompanyRepository
from app.controller.stock_controller import StockController
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class InputQueryProcessor:
    def __init__(self, db: Session):
        self.db = db
        self.stock_controller = StockController()
        self.company_repo = CompanyRepository(db)
        self.company_symbol_map = self.company_repo.get_company_symbol_map()
        self.prompt_examples = self.load_examples()
        self.api_key = os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=self.api_key)

    def load_examples(self) -> str:
        # Load prompt template
        example_path = r"D:\silverbird\query.txt"  # Adjust as needed
        try:
            with open(example_path, "r", encoding="utf-8") as file:
                return file.read()
        except FileNotFoundError:
            return ""

    def query_llm(self, prompt: str, model: str = "llama-3.3-70b-versatile") -> str:
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"Groq LLM request failed: {e}")

    def extract_info_from_query(self, query: str) -> dict:
        full_prompt = f"""
        {self.prompt_examples}

        Query: {query}
        Output (in JSON only, no explanation, no table, no formatting):
        """

        try:
            llm_response = self.query_llm(full_prompt)
            json_data = json.loads(llm_response.strip())
            print(json_data)
            return json_data
        except Exception as e:
            return {
                "error": f"Failed to extract structured data: {e}",
                "raw_response": llm_response if 'llm_response' in locals() else None
            }

    def resolve_symbol(self, user_input: str):
        match = self.company_repo.get_symbol_by_company_or_symbol(user_input)

        if isinstance(match, str):
            return match
        elif isinstance(match, dict):
            return match
        else:
            return {
                "error": "Company not found",
                "available_symbols": self.company_repo.get_all_symbols()
            }
