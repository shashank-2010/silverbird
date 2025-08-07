import os
import sys
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
grandparent_dir = os.path.abspath(os.path.join(parent_dir, '..'))

for path in [parent_dir, grandparent_dir, os.getcwd()]:
    if path not in sys.path:
        sys.path.append(path)

import csv
from fastapi import HTTPException
from sqlalchemy.orm import Session
from app.models.company import Nifty50Company

class CompaniesController:
    def __init__(self, csv_file_path: str):
        self.csv_file_path = csv_file_path

    def upload_nifty50_csv(self, db: Session):
        path = self.csv_file_path

        if not os.path.isfile(path):
            raise HTTPException(status_code=400, detail="CSV file path is invalid or file does not exist")

        records_added = 0
        with open(path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if db.query(Nifty50Company).filter_by(symbol=row["Symbol"]).first():
                    continue

                company = Nifty50Company(
                    company_name=row.get("Company Name") or row.get("company_name"),
                    industry=row.get("Industry") or row.get("industry"),
                    symbol=row.get("Symbol") or row.get("symbol"),
                    series=row.get("Series") or row.get("series"),
                    isin_code=row.get("ISIN Code") or row.get("isin_code"),
                )
                db.add(company)
                records_added += 1

            db.commit()

        return {"message": f"CSV processed, {records_added} records added successfully."}
