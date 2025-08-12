import os
import sys
import json
import random
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
from app.Utility.utility import log_execution
from app.service.InputQueryProcessor import InputQueryProcessor

load_dotenv()

class OutputQueryProcessing:
    def __init__(self, db: Session):
        self.db = db
        self.api_key = os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=self.api_key)
        self.input_query_processor = InputQueryProcessor(self.db)
        self.model = "llama-3.3-70b-versatile"

    @log_execution
    def summarize_with_llm_for_history(self, stock_data: list[dict], user_query: str) -> dict:
        if not stock_data:
            available_symbols = self.input_query_processor.company_symbol_map
            message_payload = {
            "message": "No stock data available. We are currently in beta phase.",
            "available_symbols": list(available_symbols.keys())
            }
            return self.format_info_with_llm(message_payload, user_query)

        prompt = f"""
        You are a helpful assistant. A user asked: "{user_query}"

        You are given stock data in JSON format for one or more stocks. Your task is to:
        1. Summarize the stock information clearly and concisely in plain English.
        2. Use bullet points to present key values like Open, High, Low, and Close prices.
        3. If the stock data does not match the company in the user query, note the mismatch clearly.
        4. Be conversational and avoid overly technical language.
        5. Also at last give suggestion in a polite and better way based on query like Do you want me to tell you about other symbols?
        JSON Data:
        {json.dumps(stock_data, indent=2)}
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                stream=False  # or True if you want streaming
            )
            return {"response": response.choices[0].message.content}
        except Exception as e:
            return {
                "error": f"LLM formatting failed: {e}",
                "raw_data": stock_data
            }
        
    @log_execution
    def summarize_with_llm_for_future(self, stock_data: list[dict], user_query: str) -> dict:
        if not stock_data:
            available_symbols = self.input_query_processor.company_symbol_map
            message_payload = {
            "message": "No stock data available. We are currently in beta phase.",
            "available_symbols": list(available_symbols.keys())
            }
            return self.format_info_with_llm(message_payload, user_query)

        prompt = f"""
        You are a helpful assistant. A user asked: "{user_query}"

        You are given the following stock data in JSON format. Your task is to:
        - Write a clear, concise, and short summary of the stock data.
        - Format the summary as readable bullet points. But dont mention the meta level comments like "in readable bullet form".
        - If json has direction , then try explaining the direction which is based on last day's close price. 
        - Also at last give suggestion in a polite and better way based on query like - Do you want me to let you know about other symbol?
        - If the today's date and date in stock_data is different. Then tell the user that the predicted data is from past and it can be used as backtesting.
        JSON Data:
        {json.dumps(stock_data, indent=2)}
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                stream=False  # or True if you want streaming
            )
            return {"response": response.choices[0].message.content}
        except Exception as e:
            return {
                "error": f"LLM formatting failed: {e}",
                "raw_data": stock_data
            }
    
    @log_execution
    def format_info_with_llm(self, info_payload: dict, user_query: str) -> dict:
        prompt = f"""
        A user asked: "{user_query}"

        You are given the following JSON containing an informational message and a list of available stock symbols.

        Your task is to:
        1. Rephrase the message in a polite, human-friendly tone.
        2. Present any lists (like available stock symbols) using bullet points.
        3. Keep it concise and readable.

        Respond in markdown-like format with clear headings or bullet points.

        JSON:
        {json.dumps(info_payload, indent=2)}
        """

        try:
            stream_response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a friendly assistant who clearly formats system or error messages for end-users."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                stream=True
            )

            collected_response = ""
            for chunk in stream_response:
                if chunk.choices and chunk.choices[0].delta:
                    delta = chunk.choices[0].delta
                    collected_response += delta.content or ""

            return {"response": collected_response}

        except Exception as e:
            return {
                "error": f"LLM formatting failed: {e}",
                "raw_info": info_payload
            }


    @log_execution
    def summarize_with_llm_for_no_intent(self, user_query: str) -> dict:
        user_input_lower = user_query.strip().lower()

        if user_input_lower in {"hi", "hello", "hey", "good morning", "good evening", "greetings"}:
            greeting_message = """
            Hello! 👋 I’m **SilverBird**, your AI assistant for stock analysis and predictions.

            Here’s what I can help you with:
            - 📈 Get current or historical stock prices.
            - 🔮 Predict stock prices for the next few days.
            - 📊 Analyze bullish/bearish market trends.
            - 🧠 Interpret your queries and give intelligent suggestions.

            Try asking me something like:
            1. What’s the price of TATAMOTORS today?
            2. Predict AXISBANK stock for the next 3 days.
            3. Market Trend for A Symbol
            """
            prompt = (
            "Rewrite the following assistant greeting to make it more friendly, elegant, and slightly witty. "
            "Use emojis where appropriate, keep it clear and professional, and structure it nicely:\n\n"
            + greeting_message
            )   
            response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a query rewriting assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7  # You can bump it up slightly for creativity
            )
            improved_greeting = response.choices[0].message.content.strip()

            return {"response": improved_greeting}

        # 2. Off-topic detection using simple keyword checks
        off_topic_keywords = [
            "weather", "recipe", "football", "cricket", "movie", "joke", "travel", "restaurant","country","capital"
            "love", "dating", "news", "whatsapp", "who are you", "your name", "what can you do"
        ]
        if any(kw in user_input_lower for kw in off_topic_keywords):
            message = """
                Hi there! I'm **SilverBird**, an AI designed specifically for **stock market insights and predictions for Indian Market**.

                I’m great at:
                - Checking stock prices
                - Analyzing trends
                - Predicting short-term movements
                - Giving insights on companies like Apple, Tesla, Google, etc.

                I may not be able to answer questions about other topics like movies, sports, or food. 😊

                Here are some things you can try:
                1. What’s the trend of the stock market today?
                2. Predict the stock price of Nvidia for next week.
                3. Get historical price of Microsoft for last month.
                        """
            return {"response": message.strip()}

        # 3. Default fallback using prompt examples
        prompt = f"""
            You are a helpful assistant named SilverBird who specializes in stock-related questions.

            The user asked: "{user_query}"

            Unfortunately, the system could not understand the user's intent.

            You are provided with a list of example queries below. Your task:
            - Select 5 diverse and useful examples.
            - Present them in a polite, encouraging tone.
            - Keep the examples natural and related to stocks.

            Example queries:
            {self.input_query_processor.prompt_examples}

            Respond like:
            "Sorry, I couldn’t understand your query. Here are some things you can ask me:
            1. ...
            2. ...
            3. ..."
                """

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=300,
                stream=True
            )

            full_response = ""
            for chunk in stream:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    full_response += delta.content

            return {"response": full_response.strip()}

        except Exception as e:
            return {
                "error": f"LLM streaming failed: {e}"
            }

    def rewrite_query_with_symbol(self, original_query: str, company_name: str, symbol: str) -> str:
        prompt = f"""
            You are a helpful assistant for a stock query system. Your task is to rewrite the user query using the confirmed stock symbol.

            Original query: "{original_query}"
            Resolved company name: "{company_name}"
            Resolved symbol: "{symbol}"

            Rewrite the query by replacing the ambiguous company name with the symbol '{symbol}' to make the query precise.

            Only return the updated query without explanations.
            """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a query rewriting assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLM rewrite failed: {e}")
            # Fallback: just append symbol
            return f"{original_query.strip()} ({symbol})"
        
    def format_info_with_llm_for_symbols(self, user_query: str) -> dict:
        available_symbols_map = self.input_query_processor.company_symbol_map

        sample_size = min(len(available_symbols_map), random.randint(5, 10))
        sampled_items = random.sample(list(available_symbols_map.items()), sample_size)

        formatted_samples = [f"{company} ({symbol})" for company, symbol in sampled_items]
        symbols_str = ", ".join(formatted_samples)

        prompt = f"""
        You are a helpful assistant for a stock query system.

        A user asked: "{user_query}"

        Respond politely with the following:
        - Inform the user that currently only a limited set of Nifty50 stocks are supported.
        - Encourage the user to keep exploring.
        - Mention that support for more symbols is coming soon.
        - Give a few sample companies and their symbols from the supported list, such as:
        {symbols_str}

        Keep the tone polite, informative, and user-friendly.
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant responding to a user asking about supported stock symbols."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            return {"response": response.choices[0].message.content.strip()}
        except Exception as e:
            return {
                "error": f"LLM formatting failed: {e}",
                "symbols_sampled": symbols_str
            }

    def summarize_conversation(self, history:str):
        prompt = f"""
                You are witty and helpful assistant for a stock query system.property
                Conversation history:
                {history if history.strip() else "[No Conversation History available]"}
                Task:
                - If there is no conversation history, respond humorously, for example:
                "If there’s no history, how can I provide you one? Even historians can’t do that!"
                or something playful in your own words.
                - If there is history, summarize the main points briefly and clearly.
                - Highlight any stock symbols, predictions, or key topics discussed.
                """
        try:
            response = self.client.chat.completions.create(
                model = self.model,
                messages=[
                    {"role": "system", "content": "You are a witty assistant that summarizes conversations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5
            )
            return {"response": response.choices[0].message.content.strip()}
        except Exception as e:
            return {
                "error": f"LLM summarization failed: {e}"
            }

