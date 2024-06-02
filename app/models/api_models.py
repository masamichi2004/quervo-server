from pydantic import BaseModel
class Request(BaseModel):
    location: str
    prompt:  str

class Response(BaseModel):
    id: int
    name: str
    long: float
    lat: float
    area: str
    category: str