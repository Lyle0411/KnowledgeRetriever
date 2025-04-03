from fastapi.requests import Request
from fastapi.responses import JSONResponse
from fastapi import Body, Depends, UploadFile

from typing import Optional

from ..handlers import DataPlane
from ..schemas import input_schemas
import json


class Endpoints(object):
    """
    Implementation of REST endpoints.
    These take care of the REST/HTTP-specific things and then delegate the
    business logic to the internal handlers.
    """

    def __init__(self, data_plane: DataPlane):
        self._data_plane = data_plane

    async def knowledgeRetriever(
            self, chat_vo: input_schemas.TemplateIn1 = Body()
    ) -> JSONResponse:
        result = await self._data_plane.knowledgeRetriever(chat_vo)

        return JSONResponse(content=result.dict())