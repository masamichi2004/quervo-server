from pydantic import BaseModel
from models.coordinate import Coordinate
class Prompt(BaseModel):
    current_coodinate: tuple[float, float]     # type: Coordinate
    prompt:  str    # type: str   ～したいなどの文字列
