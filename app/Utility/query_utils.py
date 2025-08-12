import os
import sys
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
from app.service.OutputQueryProcessor import OutputQueryProcessing
from app.service.InputQueryProcessor import InputQueryProcessor
from app.controller.stock_controller import StockController

class QueryUtility:
    def __init__(self, db:Session):
        self.db = db
        self.output_query = OutputQueryProcessing(self.db)
        self.input_query_processor = InputQueryProcessor(self.db)
        self.stock_controller = StockController()


    def _is_greeting(self, text):
        return text in {"hi", "hello", "hey", "greetings", "good morning", "good evening"}

    def _is_off_topic(self, text):
        off_topic_keywords = {
            "weather", "recipe", "football", "cricket", "movie", "joke", "travel",
            "restaurant", "love", "dating", "news", "whatsapp", "who are you",
            "your name", "what can you do"
        }
        return any(kw in text for kw in off_topic_keywords)

    def _valid_intents(self):
        return {
            "get_current_price",
            "get_historical_price",
            "get_future_prediction",
            "get_market_trend"
        }
    
    def _dispatch_intent_handler(self, intent, extracted, user_query, symbol, date = None):
        match intent:
            case "get_current_price":
                return self._handle_current_price(symbol, user_query)
            case "get_historical_price":
                return self._handle_historical_price(symbol, extracted, user_query)
            case "get_future_prediction":
                date = extracted.get('date',None)
                return self._handle_future_prediction(symbol, extracted, user_query, date)
            case "get_market_trend":
                return self._handle_market_trend(extracted, user_query)
            case _:
                return self.output_query.summarize_with_llm_for_no_intent(user_query)

    def _handle_current_price(self, symbol, user_query):
        raw_data = self.input_query_processor.stock_controller.get_price_data(symbol=symbol)
        return self.output_query.summarize_with_llm_for_history(raw_data, user_query)

    def _handle_historical_price(self, symbol, extracted, user_query):
        start_date = extracted.get("start_date")
        end_date = extracted.get("end_date")
        duration = extracted.get("duration")

        if duration == "1week":
            end_date = datetime.today().date().isoformat()
            start_date = (datetime.today() - timedelta(days=7)).date().isoformat()
        elif duration == "1month":
            end_date = datetime.today().date().isoformat()
            start_date = (datetime.today() - timedelta(days=30)).date().isoformat()

        if not start_date:
            start_date = (datetime.today() - timedelta(days=30)).date().isoformat()

        raw_data = self.stock_controller.get_price_data(symbol=symbol, start_date=start_date, end_date=end_date)
        return self.output_query.summarize_with_llm_for_history(raw_data, user_query)

    def _handle_future_prediction(self, symbol, extracted, user_query, date = None):
        days = extracted.get("days", 1)
        if not days:
            days = [1, 3]
        elif isinstance(days, int):
            days = [days]
        elif isinstance(days, str):
            try:
                days = [int(day.strip()) for day in days.split(",")]
            except Exception:
                days = [1]

        if date:
            prediction_data = self.stock_controller.get_past_predictions(symbol=symbol, days_list=days, date=date)
        else:
            prediction_data = self.stock_controller.get_predictions(symbol=symbol, days_list=days)
        if prediction_data.get('message'):
            return self.output_query.summarize_with_llm_for_future([prediction_data], user_query)
        processed_prediction = self.stock_controller.process_prediction_summary(prediction_data)
        return self.output_query.summarize_with_llm_for_future([processed_prediction], user_query)

    def _handle_market_trend(self, extracted, user_query):
        days = extracted.get("days", [1])
        if isinstance(days, int):
            days = [days]
        elif isinstance(days, str):
            try:
                days = [int(day.strip()) for day in days.split(",")]
            except Exception:
                days = [1]

        direction = extracted.get("direction", "up")
        trends = self.stock_controller.get_market_trends(direction=direction, days=days)
        return self.output_query.summarize_with_llm_for_future(trends, user_query)

