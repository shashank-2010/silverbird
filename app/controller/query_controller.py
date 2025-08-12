import os
import sys
import json
import re
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
from app.Utility.redis_session_context import RedisSessionContextManager
from app.feature_engineering.save_features_csv import StoreFeatureEngineering

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
        self.SessionManager = RedisSessionContextManager()
        self.comp_repo = CompanyRepository(db=self.db)

    def _resolve_symbol_with_context(self, extracted, user_query, session_id):
        input_company = extracted.get("company")
        try:
            resolved = self.input_query_processor.resolve_symbol(input_company)
            if isinstance(resolved, dict) and "error" in resolved:
                return self.output_query.format_info_with_llm(resolved, input_company)
            if isinstance(resolved, dict) and "suggestion" in resolved:
                self.SessionManager.save_context(session_id, {
                    "original_query": user_query,
                    "symbol": resolved["symbol"],
                    "intent": extracted["intent"]
                })
                return resolved
            return resolved
        finally:
            self.db.close()
    
    def _handle_confirmation(self, session_id, symbol = None):
        extracted = self.SessionManager.get_context(session_id)
        if extracted:
            self.SessionManager.clear_context(session_id)
            original_query = extracted["original_query"]
            symbol = symbol
            company_name = extracted.get("company_name", "")  # optional

            rewritten_query = self.output_query.rewrite_query_with_symbol(
                original_query=original_query,
                company_name=company_name,
                symbol=symbol
            )
            return self.process_user_query(rewritten_query, session_id)
        return {"error": "No previous suggestion found to confirm."}
    
    def _normalize_input(self, input_str: str) -> str:
        return re.sub(r'[^\w\s]', '', input_str.strip().lower())
    
    def _handle_technical_analysis_query(self, user_query, user_input_lower, session_id, extracted):
        symbol = None
        if "company" in extracted:
            symbol = self.comp_repo.get_symbol_by_company_or_symbol(extracted["company"])
        else:
            ctx = self.SessionManager.get_context(session_id)
            if ctx and "symbol" in ctx:
                symbol = ctx["symbol"]

        if not symbol:
            reply = {"message": "Please provide a company name or symbol for indicator analysis."}
            return self.output_query.format_info_with_llm(info_payload=reply, user_query=user_input_lower)

        try:
            self.stored_enriched_csv = StoreFeatureEngineering(symbol=symbol)
            enriched_csv_path = self.stored_enriched_csv._generate_enriched_csv_with_features()

            reply = self.output_query.summarize_ta_using_llm(
                enriched_csv_path=enriched_csv_path,
                user_query=user_query
            )

            self.SessionManager.save_context(session_id, {
                "symbol": symbol,
                "intent": "technical_indicator_query",
                "original_query": user_input_lower
            })

            self.SessionManager.save_llm_response(session_id, "assistant", reply)
            return reply

        except Exception as e:
            return {"error": str(e)}


    @log_execution
    def process_user_query(self, user_query: str, session_id: str):
        try:
            self.SessionManager.save_message(session_id, "user", user_query)
            user_input_lower = user_query.strip().lower()
            extracted = self.input_query_processor.extract_info_from_query(user_input_lower)
            confirmation = extracted.get("confirmation", "").lower()

            if "summary" in user_input_lower or "previous" in user_input_lower:
                history = self.SessionManager.get_all_messages(session_id)
                reply = self.output_query.summarize_conversation(history)
                self.SessionManager.save_llm_response(session_id, "assistant", reply)
                return reply

            # 1. Handle greetings / off-topic
            if self.query_util._is_greeting(user_input_lower) or self.query_util._is_off_topic(user_input_lower):
                reply = self.output_query.summarize_with_llm_for_no_intent(user_query)
                self.SessionManager.save_llm_response(session_id, "assistant", reply)
                return reply

            # 2. Handle confirmations
            confirm_words = {
                "yes", "y", "yeah", "correct", "right", "yep", "sure",
                "absolutely", "ok", "okay", "affirmative"
            }
            if any(word in user_input_lower.split() for word in confirm_words) or confirmation =='yes':
                company = extracted.get('company','')
                if company:
                    symbol = self.comp_repo.get_symbol_by_company_or_symbol(company)
                    return self._handle_confirmation(session_id, symbol = symbol)
                reply = self._handle_confirmation(session_id, symbol=None)
                self.SessionManager.save_llm_response(session_id, "assistant", reply)
                return self.output_query.format_info_with_llm(info_payload=reply, user_query=user_input_lower)

            # 3. Handle negatives
            negative_words = {"no", "n", "wrong", "nopes", "nah", "not at all"}
            if any(word in user_input_lower.split() for word in negative_words) or confirmation =='no':
                self.SessionManager.clear_context(session_id)
                reply = {"message": "Okay, please provide the correct company name or symbol."}
                self.SessionManager.save_llm_response(session_id, "assistant", reply)
                return self.output_query.format_info_with_llm(info_payload=reply, user_query=user_input_lower)

            # 4. Extract info from query
            # extracted = self.input_query_processor.extract_info_from_query(user_query)
            if extracted.get("intent") == "symbols from database":
                return self.output_query.format_info_with_llm_for_symbols(user_query)
            
            if extracted.get("intent") == "technical_indicator_query":
                return self._handle_technical_analysis_query(user_query, user_input_lower, session_id, extracted)



            # 5. Handle company-change follow-ups if intent unknown
            if extracted.get("intent") == "unknown":
                change_triggers = ["tell me about", "now tell me about", "show me", "about", "regarding"]
                company_part = None
                for trigger in change_triggers:
                    if trigger in user_input_lower:
                        company_part = user_input_lower.split(trigger, 1)[-1].strip()
                        break
                # Also handle if user just says "TCS", "Reliance", etc.
                if not company_part and len(user_input_lower.split()) <= 3:
                    blacklist_keywords = ["capital", "population", "president", "weather", "prime minister", "city"]
                    if not any(word in user_input_lower for word in blacklist_keywords):
                        company_part = user_input_lower

                if company_part:
                    symbol_data = self.input_query_processor.resolve_symbol(company_part)
                    if symbol_data:
                        last_ctx = self.SessionManager.get_context(session_id)
                        if last_ctx and "intent" in last_ctx:
                            intent = last_ctx["intent"]
                            symbol = symbol_data["symbol"] if isinstance(symbol_data, dict) else symbol_data
                            reply = self.query_util._dispatch_intent_handler(intent, {"company": company_part}, user_query, symbol, date =None)
                            self.SessionManager.save_context(session_id, {
                                "symbol": symbol,
                                "intent": intent,
                                "original_query":user_input_lower
                            })
                            self.SessionManager.save_llm_response(session_id, "assistant", reply)
                            return reply
                else:
                    return self.output_query.summarize_with_llm_for_no_intent(user_query=user_query)

            # 6. If extraction failed
            if "error" in extracted or "intent" not in extracted:
                reply = self.output_query.summarize_with_llm_for_no_intent(user_query)
                self.SessionManager.save_llm_response(session_id, "assistant", reply)
                return reply

            intent = extracted["intent"]
            if intent not in self.query_util._valid_intents():
                reply = self.output_query.summarize_with_llm_for_no_intent(user_query)
                self.SessionManager.save_message(session_id, "assistant", reply)
                return reply

            # 7. Symbol resolution
            symbol = None
            if intent != "get_market_trend":
                ctx = self.SessionManager.get_context(session_id)
                if ctx and "symbol" in ctx and "company" not in extracted:
                    symbol = ctx["symbol"]  # Use previous symbol
                else:
                    symbol = self._resolve_symbol_with_context(extracted, user_query, session_id)
                    if isinstance(symbol, dict):  # Suggestion case
                        self.SessionManager.save_llm_response(session_id, "assistant", symbol)
                        return self.output_query.format_info_with_llm(info_payload=symbol, user_query=user_input_lower)

            # 8. Handle main intent
            reply = self.query_util._dispatch_intent_handler(intent, extracted, user_query, symbol, date=None)

            # 9. Save context
            self.SessionManager.save_context(session_id, {
                "symbol": symbol,
                "intent": intent,
                "original_query":user_input_lower
            })
            self.SessionManager.save_llm_response(session_id, "assistant", reply)

            return reply

        except Exception as e:
            return {"error": f"Unexpected error while processing query: {str(e)}"}
