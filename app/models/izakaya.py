from pydantic import BaseModel
from typing import Optional
class Izakaya(BaseModel):
    id: int
    name: str
    lng: float
    lat: float
    area: str
    distance: Optional[float] = None
    category: str