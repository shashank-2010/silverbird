import os
from dotenv import load_dotenv
load_dotenv()
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME")

POSTGRES_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}"
#POSTGRES_URL = "postgresql://postgres:12345678@localhost:5432/silverbird"

engine = create_engine(POSTGRES_URL)
SessionLocal = sessionmaker(bind=engine,autoflush=False,autocommit = False)
Base = declarative_base()

def init_db():
    from app.models import company
    Base.metadata.create_all(bind=engine)