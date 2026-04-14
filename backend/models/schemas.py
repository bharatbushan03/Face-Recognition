from pydantic import BaseModel, ConfigDict
from typing import Optional, List
import datetime

class UserBase(BaseModel):
    name: str

class UserResponse(UserBase):
    id: int
    created_at: datetime.datetime
    
    model_config = ConfigDict(from_attributes=True)

class BaseResponse(BaseModel):
    status: str
    message: str

class ErrorResponse(BaseModel):
    status: str = "error"
    message: str
    code: int

class RegisterResponse(BaseModel):
    status: str = "success"
    message: str
    user: UserResponse

class RecognizeResponse(BaseModel):
    status: str = "success"
    message: str
    match_found: bool
    user: Optional[UserResponse] = None
    confidence: Optional[float] = None
    box: Optional[List[int]] = None
    is_smiling: Optional[bool] = None

class Base64ImageRequest(BaseModel):
    image_base64: str
    name: Optional[str] = None
