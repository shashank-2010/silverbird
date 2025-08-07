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

from fastapi import HTTPException
from sqlalchemy.orm import Session
from app.service.UserService import UserServices
from app.schemas.users import UserCreate, UserLogin, DeactivateAcc, ModifyUser
from app.auth.jwt_handler import create_access_token
from app.models.users import Users

class UserController:
    def __init__(self, db: Session):
        self.db = db
        self.service = UserServices(db)


    def signup(self, user_data: UserCreate):
        try:
            user = self.service.create_user(user_data)
            if not user:
                return {"msg": "User already registered"}
            return {"msg": "Signup successful", "user_id": user_data.user_name}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))


    def login(self, credentials: UserLogin):
        user = self.service.authenticate_user(username=credentials.user_name,useremail=None,password=credentials.password)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid username or password")

        token = create_access_token({"sub": user.user_name})
        return {"access_token": token, "token_type": "bearer"}


    def logout(self):
        # Optional: implement JWT blacklisting for advanced logout
        return {"msg": "Logged out"}
    

    def deactivate_acc(self, current_user: DeactivateAcc):
        if not current_user:
            return {"error": "User not authenticated"}
        user = self.service.authenticate_user(username=None,useremail=current_user.user_email, password=current_user.password)
        if user:
            self.service.deactivate_account(current_user.user_id)
            return {"message": "Account successfully deactivated"}
        else:
            return {"message": "Something went wrong. User can't be authenticated"}


    def delete_account(self, current_user: DeactivateAcc):
        if not current_user:
            return {"error": "User not authenticated"}
        user = self.service.authenticate_user(username=None,useremail=current_user.user_email, password=current_user.password)
        if user:
            self.service.delete_account(current_user.user_id)
            return {"message": "Account successfully removed."}
        else:
            return {"message": "Something went wrong. User can't be authenticated"}
    
    def modify_user_details(self, user: ModifyUser):
        if not user or not user.user_email_address:
            raise HTTPException(status_code=400, detail="User data or email missing")

        self.service.modify_user_details(user_data=user)
        return {"msg": "Profile Updated"}
