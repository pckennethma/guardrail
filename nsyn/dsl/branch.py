from nsyn.dsl.assign import DSLAssign
from nsyn.dsl.condition import DSLCondition
from nsyn.dsl.util import get_keyword_text
from nsyn.util.base_model import BaseModel

IF_str = get_keyword_text("IF")
THEN_str = get_keyword_text("THEN")


class DSLBranch(BaseModel):
    condition: DSLCondition
    assign: DSLAssign

    def __str__(self) -> str:
        return f"{IF_str} {self.condition} {THEN_str} {self.assign}"
