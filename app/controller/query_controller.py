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

    # @log_execution
    # def process_user_query(self, user_query: str):
    #     extracted = self.input_query_processor.extract_info_from_query(user_query)

    #     if "error" in extracted or "intent" not in extracted:
    #         return extracted if "error" in extracted else {
    #             "error": "Could not extract necessary data from query."
    #         }

    #     intent = extracted["intent"]
    #     input_company = extracted["company"] if intent != 'get_market_trend' else None

    #     try:
    #         if input_company:
    #             resolved = self.input_query_processor.resolve_symbol(input_company)
    #             if isinstance(resolved, dict):
    #                 return resolved

    #             symbol = resolved 
    #     finally:
    #         self.db.close()

    #     if intent == "get_current_price":
    #         raw_data = self.input_query_processor.stock_controller.get_price_data(symbol=symbol)
    #         return OutputQueryProcessing.summarize_with_llm_for_history(raw_data, user_query)

    #     elif intent == "get_historical_price":
    #         start_date = extracted.get("start_date")
    #         end_date = extracted.get("end_date")
    #         duration = extracted.get("duration")

    #         if duration == "1week":
    #             end_date = datetime.today().date().isoformat()
    #             start_date = (datetime.today() - timedelta(days=7)).date().isoformat()
    #         elif duration == "1month":
    #             end_date = datetime.today().date().isoformat()
    #             start_date = (datetime.today() - timedelta(days=30)).date().isoformat()

    #         if not start_date:
    #             start_date = (datetime.today() - timedelta(days=30)).date().isoformat()

    #         raw_data = self.stock_controller.get_price_data(
    #             symbol=symbol,
    #             start_date=start_date,
    #             end_date=end_date
    #         )
    #         return OutputQueryProcessing.summarize_with_llm_for_history(raw_data, user_query)
        
    #     elif intent == "get_future_prediction":
    #         days = extracted.get("days")
    #         if not days:
    #             days = [1,3]
    #         elif isinstance(days, int):
    #             days = [days]
    #         elif isinstance(days, str):
    #             try:
    #                 days = [int(day.strip()) for day in days.split(",")]
    #             except Exception:
    #                 days = [1]

    #         prediction_data = self.stock_controller.get_predictions(symbol=symbol, days_list=days)
    #         processed_prediction = self.stock_controller.process_prediction_summary(prediction_data)
    #         return OutputQueryProcessing.summarize_with_llm_for_future([processed_prediction], user_query)
        
    #     elif intent == "get_market_trend":
    #         days = extracted.get("days", [1]) 
    #         if isinstance(days, int):
    #             days = [days]
    #         elif isinstance(days, str):
    #             try:
    #                 days = [int(day.strip()) for day in days.split(",")]
    #             except Exception:
    #                 days = [1]

    #         direction = extracted.get("direction", "up")

    #         trends = self.stock_controller.get_market_trends(direction=direction, days=days)
    #         return OutputQueryProcessing.summarize_with_llm_for_future(trends, user_query)
    #     else:
    #         return OutputQueryProcessing.summarize_with_llm_for_no_intent(user_query)


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
