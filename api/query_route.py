from fastapi import APIRouter, Query
from app.database import SessionLocal
from app.controller.query_controller import QueryController

router = APIRouter(prefix="/query", tags=["Query"])

from contextlib import contextmanager

@contextmanager
def get_db_session():
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

with get_db_session() as db:
    query_controller = QueryController(db=db)

@router.get("/process_query")
def process_query(user_query: str = Query(..., description="User's natural language query"),
                  user_id: str = Query(..., description="Unique user ID for session tracking")):
    return query_controller.process_user_query(user_query, session_id=user_id)
