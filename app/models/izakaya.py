from pydantic import BaseModel
class Izakaya(BaseModel):
    id: int
    name: str
    long: float
    lat: float
    area: str
    category: str