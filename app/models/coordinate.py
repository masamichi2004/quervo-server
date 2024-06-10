from pydantic import BaseModel
class Coordinate(BaseModel):
    coordinate: tuple[float, float]     # tuple[float, float] = [lat, lng]
