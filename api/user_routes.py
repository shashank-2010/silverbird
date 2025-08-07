# user_routes.py
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from app.controller.user_controller import UserController
from app.schemas.users import UserCreate, UserLogin, ModifyUser
from app.database import SessionLocal
from app.models.users import Users
from app.auth.jwt_handler import get_current_user

router = APIRouter(prefix="/users", tags=["Users"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/signup")
def user_signup(user: UserCreate, db: Session = Depends(get_db)):
    return UserController(db).signup(user)

@router.post("/login")
def login(data: UserLogin, db: Session = Depends(get_db)):
    return UserController(db).login(data)

@router.post("/logout")
def logout(db: Session = Depends(get_db)):
    return UserController(db).logout()

@router.get("/secure/profile")
def get_profile(current_user: Users = Depends(get_current_user)):
    return {
        "user_id": current_user.user_id,
        "username": current_user.user_name,
        "email": current_user.user_email_address,
        "role_uuid": current_user.role_uuid
    }

@router.post("/deactivate_account")
def deactivate(current_user: Users = Depends(get_current_user), db: Session = Depends(get_db)):
    return UserController(db).deactivate_acc(current_user)

@router.post("/delete_account")
def delete_account(current_user: Users = Depends(get_current_user), db: Session = Depends(get_db)):
    return UserController(db).delete_account(current_user)

@router.post("/update_user_info")
def update_account(current_user: ModifyUser, db: Session = Depends(get_db)):
    return UserController(db).modify_user_details(current_user)
