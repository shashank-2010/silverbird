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

from sqlalchemy import Column, String, Integer, DateTime, Boolean, UniqueConstraint
from sqlalchemy import Enum as SQLEnum
from sqlalchemy.sql import func
from app.Utility.enums import UserStatusEnum
import uuid
from app.database import Base

class Users(Base):
    __tablename__ = 'users'
    __tableargs__ = (UniqueConstraint('user_name', name = 'uq_user_name'),)

    #id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    user_uuid = Column(Integer, unique=True, index=True)
    role_uuid = Column(Integer, default=1)

    user_first_name = Column(String, nullable=False)
    user_last_name = Column(String)
    user_email_address = Column(String, nullable=False)
    user_phone_number = Column(String)
    user_birth_date = Column(String)

    city = Column(String, nullable=False)
    county = Column(String, nullable=False)
    state = Column(String, nullable=False)

    status = Column(SQLEnum(UserStatusEnum), default=UserStatusEnum.active, nullable=False)
    user_gender = Column(String)
    user_image = Column(String)

    password = Column(String, nullable=False)

    created_tms = Column(DateTime(timezone=True), server_default=func.now())
    modified_tms = Column(DateTime(timezone=True), onupdate=func.now())

    user_name = Column(String, unique=True, nullable=False)