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
from app.service.InputQueryProcessor import InputQueryProcessor
from app.controller.stock_controller import StockController
from app.Utility.utility import log_execution
from app.service.OutputQueryProcessor import OutputQueryProcessing
from app.Utility.session_context import SessionContextManager
from app.Utility.query_utils import QueryUtility

import openai
from dotenv import load_dotenv

load_dotenv()

# openai.api_key = os.getenv("GROQ_API_KEY")
# openai.api_base = "https://api.groq.com/openai/v1"


class QueryController:
    def __init__(self, db: Session):
        self.db = db
        self.stock_controller = StockController()
        self.input_query_processor = InputQueryProcessor(self.db)
        self.api_key = os.getenv("GROQ_API_KEY")
        #self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.output_query = OutputQueryProcessing(self.db)
        self.query_util = QueryUtility(self.db)

    def _resolve_symbol_with_context(self, extracted, user_query, session_id):
        input_company = extracted.get("company")
        try:
            resolved = self.input_query_processor.resolve_symbol(input_company)
            if isinstance(resolved, dict) and "suggestion" in resolved:
                SessionContextManager.save_context(session_id, {
                    "original_query": user_query,
                    "symbol": resolved["symbol"],
                    "intent": extracted["intent"]
                })
                return resolved
            return resolved
        finally:
            self.db.close()
    
    def _handle_confirmation(self, session_id):
        extracted = SessionContextManager.get_context(session_id)
        if extracted:
            SessionContextManager.clear_context(session_id)
            original_query = extracted["original_query"]
            symbol = extracted["symbol"]
            company_name = extracted.get("company_name", "")  # optional

            rewritten_query = self.output_query.rewrite_query_with_symbol(
                original_query=original_query,
                company_name=company_name,
                symbol=symbol
            )
            return self.process_user_query(rewritten_query, session_id)
        return {"error": "No previous suggestion found to confirm."}



    @log_execution
    def process_user_query(self, user_query: str, session_id: str = "default_session"):
        try:
            user_input_lower = user_query.strip().lower()

            if self.query_util._is_greeting(user_input_lower) or self.query_util._is_off_topic(user_input_lower):
                return self.output_query.summarize_with_llm_for_no_intent(user_query)

            if user_input_lower in {"yes", "y", "yeah", "correct", "right"}:
                return self._handle_confirmation(session_id)

            if user_input_lower in {"no", "n", "wrong", "nopes"}:
                SessionContextManager.clear_context(session_id)
                return {"message": "Okay, please provide the correct company name or symbol."}

            extracted = self.input_query_processor.extract_info_from_query(user_query)

            if "error" in extracted or "intent" not in extracted:
                return self.output_query.summarize_with_llm_for_no_intent(user_query)

            intent = extracted["intent"]
            if intent not in self.query_util._valid_intents():
                return self.output_query.summarize_with_llm_for_no_intent(user_query)

            symbol = None
            if intent != "get_market_trend":
                symbol = self._resolve_symbol_with_context(extracted, user_query, session_id)
                if isinstance(symbol, dict): 
                    return symbol

            # Delegate to appropriate intent handler
            return self.query_util._dispatch_intent_handler(intent, extracted, user_query, symbol)

        except Exception as e:
            return {"error": f"Unexpected error while processing query: {str(e)}"}
