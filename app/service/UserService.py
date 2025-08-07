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
from passlib.context import CryptContext
from app.models.users import Users
from app.schemas.users import UserCreate, DeactivateAcc, ModifyUser
from app.repository.UserRepository import UserRepository
from app.Utility.enums import UserStatusEnum
from fastapi import HTTPException
from datetime import datetime
import uuid

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class UserServices:
    def __init__(self, db:Session):
        self.db = db
        self.repo = UserRepository(db)

    def create_user(self,user_data:UserCreate):
        if self.repo.get_by_useremail(user_data.user_email_address) or self.repo.get_by_username(user_data.user_name):
            return None
        
        last_uuid = self.repo.get_last_user_uuid()
        new_uuid = (last_uuid if last_uuid else 9) + 1
        hashed_pw = pwd_context.hash(user_data.password)
        new_user = Users(
            user_id=str(uuid.uuid4()),
            user_uuid=new_uuid,
            role_uuid=1,
            user_first_name=user_data.user_first_name,
            user_last_name=user_data.user_last_name,
            user_email_address=user_data.user_email_address,
            user_phone_number=user_data.user_phone_number,
            user_birth_date=user_data.user_birth_date,
            city=user_data.city,
            county=user_data.county,
            state=user_data.state,
            user_gender=user_data.user_gender,
            user_image=user_data.user_image,
            password=hashed_pw,
            user_name=user_data.user_name
        )
        return self.repo.create(new_user)
    
    def authenticate_user(self, username:str, useremail:str, password:str):
        if username:
            user = self.repo.get_by_username(username)
        else:
            user = self.repo.get_by_useremail(useremail)

        if not user or not pwd_context.verify(password, user.password):
            return "Invalid Passowrd"
        return user
    
    def deactivate_account(self,user_id: str):
        user = self.repo.get_by_userid(user_id=user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        user.status = UserStatusEnum.inactive
        user.modified_tms = datetime.now()
        self.db.commit()
    
    def delete_account(self, user_id: str):
        user = self.repo.get_by_userid(user_id=user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        user.status = UserStatusEnum.delete
        user.modified_tms = datetime.now()
        self.db.commit()

    def modify_user_details(self, user_data: ModifyUser):
        db_user = self.repo.get_by_useremail(user_data.user_email_address)

        if not db_user:
            raise HTTPException(status_code=404, detail="User not found")

        for field, value in user_data.model_dump(exclude_unset=True).items():
            if field == "password" and value:
                setattr(db_user, field, pwd_context.hash(value))
            elif value is not None:
                setattr(db_user, field, value)

        self.db.commit()
        self.db.refresh(db_user)

        return {"msg": "User details updated successfully", "user_id": db_user.user_id}
