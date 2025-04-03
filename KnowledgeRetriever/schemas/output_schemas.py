from pydantic import BaseModel, Field
from enum import Enum, IntEnum
from typing import List


class TemplateOut1(BaseModel):
    result: dict = Field(description="相关条例字典")