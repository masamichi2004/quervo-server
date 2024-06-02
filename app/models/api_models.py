from pydantic import BaseModel
class Request(BaseModel):
    location: str
    prompt:  str
