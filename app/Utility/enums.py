from enum import Enum

class UserStatusEnum(int, Enum):
    active = 0 
    inactive = 1
    delete = 2