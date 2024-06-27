from pydantic import BaseModel
from typing import Optional
class Izakaya(BaseModel):
    id: int
    name: str
    lat: float
    lng: float
    distance: Optional[float] = None
    category: str
    photo_url: str
    address: str
    izakaya_url: str