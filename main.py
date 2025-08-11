from fastapi import FastAPI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from app.database import Base, engine, init_db
from api.user_routes import router as user_router
from api.stock_routes import router as stock_router
from api.company_routes import router as company_router
from api.query_route import router as query_router
#from api import api_route

Base.metadata.create_all(bind=engine)
init_db()

app = FastAPI(
    title="SilverBird",
    description="API for Stock Prediction and User Management",
    version="1.0.0"
)

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5500",  
    "http://localhost:5500"
]

#http://silverbirds.in/
# origins = [
#     "https://silverbirds.in",
#     "https://www.silverbirds.in",
#     "https://api.silverbirds.in"
# ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#app.include_router(api_route.router)
app.include_router(user_router)
app.include_router(stock_router)
app.include_router(company_router)
app.include_router(query_router)


