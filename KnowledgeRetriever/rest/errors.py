from typing import Optional
from fastapi import Request
from pydantic import BaseModel

from .responses import Response
from ..errors import ServerError


class APIErrorResponse(BaseModel):
    error: Optional[str] = None


async def handle_dgserver_error(request: Request, exc: ServerError) -> Response:
    err_res = APIErrorResponse(error=str(exc))
    return Response(status_code=exc.status_code, content=err_res.dict())


_EXCEPTION_HANDLERS = {ServerError: handle_dgserver_error}