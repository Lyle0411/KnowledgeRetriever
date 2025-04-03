from pydantic import BaseModel as _BaseModel, Field, ConfigDict, Extra

from typing import Optional, List, Dict, Any, Union, TextIO, BinaryIO
from io import BytesIO, TextIOBase
from pydantic.dataclasses import dataclass


class BaseModel(_BaseModel):
    model_config = ConfigDict(
        title="Entity config",
        extra=Extra.ignore,
        arbitrary_types_allowed=True
    )

"""
https://fastapi.tiangolo.com/tutorial/request-forms/#about-form-fields

As per FastAPI documentation:
You can declare multiple Form parameters in a path operation, but you can't
also declare Body fields that you expect to receive as JSON, as the request
will have the body encoded using application/x-www-form-urlencoded instead 
of application/json (when the form includes files, it is encoded as 
multipart/form-data).

This is not a limitation of FastAPI, it's part of the HTTP protocol.
"""
class TemplateIn1(_BaseModel):
    query: Optional[str] = Field(default="矿井核定生产能力90万吨/年，2024年1月生产原煤10.34万吨",
                                description="案例描述")
    
class TemplateIn2(BaseModel):
    text: Optional[str] = Field(description="输入文本")

      