from pydantic import BaseModel
class Izakaya(BaseModel):
    id: int
    name: str
    lng: float
    lat: float
    area: str
    category: str