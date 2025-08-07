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

from sqlalchemy.orm import Session
from sqlalchemy import text
from app.models.company import Nifty50Company
from difflib import get_close_matches

class CompanyRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_company_symbol_map(self):
        query = text("SELECT company_name, symbol FROM companies") 
        result = self.db.execute(query).fetchall()
        return {row[0]: row[1] for row in result}

    def get_symbol_by_company_or_symbol(self, user_input: str):
        user_input = user_input.lower()
        company_symbol_map = self.get_company_symbol_map()

        # First: Exact match by symbol
        for name, symbol in company_symbol_map.items():
            if user_input == symbol.lower():
                return symbol

        # Second: Exact match by company name
        for name, symbol in company_symbol_map.items():
            if user_input == name.lower():
                return symbol

        # Third: Fuzzy match by company name
        company_names = list(company_symbol_map.keys())
        company_matches = get_close_matches(user_input, company_names, n=1, cutoff=0.5)
        if company_matches:
            matched_name = company_matches[0]
            return {
                "suggestion": f"Did you mean '{matched_name}'?",
                "symbol": company_symbol_map[matched_name],
                "note": "Please confirm before proceeding."
            }

        # Fourth: Fuzzy match by symbol
        symbol_to_company_map = {v: k for k, v in company_symbol_map.items()}
        symbol_names = list(symbol_to_company_map.keys())
        symbol_matches = get_close_matches(user_input, symbol_names, n=1, cutoff=0.3)
        if symbol_matches:
            matched_symbol = symbol_matches[0]
            return {
                "suggestion": f"Did you mean symbol '{matched_symbol}' for '{symbol_to_company_map[matched_symbol]}'?",
                "symbol": matched_symbol,
                "note": "Please confirm before proceeding."
            }

        # Fifth: Substring match (e.g., 'adani' in 'Adani Enterprises Ltd.')
        for name, symbol in company_symbol_map.items():
            if user_input in name.lower():
                return {
                    "suggestion": f"Did you mean '{name}'?",
                    "symbol": symbol,
                    "note": "Please confirm before proceeding."
                }

        return None


    def get_all_symbols(self):
        query = text("SELECT symbol FROM companies")
        result = self.db.execute(query).fetchall()
        return [row[0] for row in result]

    def get_all_companies(self):
        return self.db.query(Nifty50Company.company_name, Nifty50Company.symbol).all()


        
    


        