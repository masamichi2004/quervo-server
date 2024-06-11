from pydantic import BaseModel
class Prompt(BaseModel):
    current_coodinate: tuple[float, float]     # type: Coordinate
    prompt:  str    # type: str   ～したいなどの文字列
