from fastapi.requests import Request
from fastapi.responses import JSONResponse
from fastapi import Body, Depends, UploadFile, File, HTTPException,Form
from typing import Optional
import os
from ..handlers import DataPlane
from ..schemas import input_schemas
import json
from pathlib import Path


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


    async def graphCreate(self,
                          subject: str = Form(...),
                          title: str = Form(...),
                          topic: str = Form(...),
                          file: UploadFile = File(...)) -> JSONResponse:

        save_dir = (
                Path(__file__).parent.parent
                / "core"
                / "data"
        )

        # 确保目录存在（mkdir 代替 os.makedirs）
        save_dir.mkdir(parents=True, exist_ok=True)  # parents=True 会自动创建中间目录

        # 构建完整保存路径（Path 对象）
        save_path = save_dir / file.filename  # 直接使用 / 运算符拼接路径

        content = await file.read()
        try:
            with open(save_path, "wb") as buffer:
                buffer.write(content)
                result = await self._data_plane.create_graph(subject, title, topic, str(save_path))
                return JSONResponse(content=result)
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})
        finally:
            if Path(save_path).exists():  # 再次显式转换确保类型
                Path(save_path).unlink()



