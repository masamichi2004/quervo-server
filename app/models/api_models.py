from pydantic import BaseModel
from typing import Optional
from app.models.coordinate import Coordinate
class Prompt(BaseModel):
    current_coodinate: Optional[Coordinate] = None     # type: Coordinate
    prompt:  str
