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
from app.models.users import Users
from app.Utility.enums import UserStatusEnum

class UserRepository:
    def __init__(self, db:Session):
        self.db = db

    def get_by_username(self, username:str):
        return self.db.query(Users).filter(Users.user_name == username).first()
    
    def get_last_user_uuid(self):
        result = (self.db.query(Users.user_uuid).order_by(Users.user_uuid.desc()).first())
        return result[0] if result else None
    
    def create(self, user:Users):
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user
    
    def get_by_useremail(self, email:str):
        return self.db.query(Users).filter(Users.user_email_address == email).first()
    
    def get_by_userid(self, user_id :str):
        return self.db.query(Users).filter(Users.user_id == user_id).first()
        
    


        