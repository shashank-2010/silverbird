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

from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime
from app.Utility.enums import UserStatusEnum

class UserCreate(BaseModel):
    user_first_name: str
    user_last_name: Optional[str] = None
    user_email_address: EmailStr
    user_phone_number: Optional[str] = None
    user_birth_date: Optional[str] = None
    city: str
    county: str
    state: str
    user_gender: Optional[str] = None
    user_image: Optional[str] = None
    password: str
    user_name: str = Field(..., pattern = r"^[A-Za-z0-9_]+$")

class UserLogin(BaseModel):
    user_name: Optional[str] = None
    user_email :Optional[str] = None
    password: str

class DeactivateAcc(BaseModel):
    user_id : str
    user_email:str
    password: str
    status: UserStatusEnum

class ModifyUser(BaseModel):
    user_first_name: Optional[str] = None
    user_last_name: Optional[str] = None
    user_email_address: EmailStr
    user_phone_number: Optional[str] = None
    user_birth_date: Optional[str] = None
    city: Optional[str] = None
    county: Optional[str] = None
    state: Optional[str] = None
    user_gender: Optional[str] = None
    user_image: Optional[str] = None
    password: Optional[str] = None


