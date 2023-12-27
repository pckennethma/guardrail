from typing import Any

from nsyn.util.base_model import BaseModel
from nsyn.util.color import get_keyword_text

ASSIGN_str = get_keyword_text("<-")


class DSLAssign(BaseModel):
    variable: str
    value: Any

    def execute(self) -> Any:
        return self.value

    def __str__(self) -> str:
        return f"{self.variable} {ASSIGN_str} {self.value}"
