from fastapi import status
from typing import Optional

from odmantic.exceptions import DocumentParsingError


# 文档生成项目父类异常
class ServerError(Exception):
    def __init__(self, msg: str, status_code: int = status.HTTP_400_BAD_REQUEST):
        super().__init__(msg)
        self.status_code = status_code


# 可扩展子类异常，以满足不同模块需求，前提是必须继承'DGServerError'。
class ExampleError(ServerError):
    def __init__(self):
        super().__init__(
            f"ExampleError", status.HTTP_400_BAD_REQUEST
        )


class MongoDocumentParsingError(ServerError):

    def __init__(self, message):
        super().__init__(
            message, status.HTTP_500_INTERNAL_SERVER_ERROR
        )


class MongoDuplicateKeyError(ServerError):

    def __init__(self, message):
        super().__init__(
            message, status.HTTP_500_INTERNAL_SERVER_ERROR
        )
