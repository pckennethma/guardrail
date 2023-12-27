from nsyn.dsl.assign import DSLAssign
from nsyn.dsl.condition import DSLCondition
from nsyn.util.base_model import BaseModel
from nsyn.util.color import get_keyword_text

IF_str = get_keyword_text("IF")
THEN_str = get_keyword_text("THEN")


class DSLBranch(BaseModel):
    condition: DSLCondition
    assign: DSLAssign

    def __str__(self) -> str:
        return f"{IF_str} {self.condition} {THEN_str} {self.assign}"
