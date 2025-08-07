# company_routes.py
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.controller.companies_controller import CompaniesController
from app.database import SessionLocal

router = APIRouter(prefix="/companies", tags=["Companies"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/upload_symbol_data")
def upload_symbol_data(csv_file_path: str, db: Session = Depends(get_db)):
    controller = CompaniesController(csv_file_path=csv_file_path)
    return controller.upload_nifty50_csv(db)
