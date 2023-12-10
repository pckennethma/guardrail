from typing import Any

from nsyn.dsl.util import get_keyword_text
from nsyn.util.base_model import BaseModel

ASSIGN_str = get_keyword_text("<-")


class DSLAssign(BaseModel):
    variable: str
    value: Any

    def execute(self) -> Any:
        return self.value

    def __str__(self) -> str:
        return f"{self.variable} {ASSIGN_str} {self.value}"
