from pydantic import BaseModel
from typing import Tuple
class Coordinate(BaseModel):
    coordinate: Tuple[float, float]     # tuple[float, float] = [lng, lat]
