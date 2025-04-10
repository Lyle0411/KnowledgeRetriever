import sys
import os
import json
import importlib
import inspect

from typing import Any, Dict, List, Optional, Type, Union, no_type_check, TYPE_CHECKING
from pydantic import PyObject, Extra, Field
from pydantic_settings import BaseSettings as _BaseSettings
from contextlib import contextmanager

from .version import __version__

ENV_FILE_SETTINGS = ".env"
# 修改ENV_PREFIX_SETTINGS为""，以让所有.env中的环境变量都能被检查到
ENV_PREFIX_SETTINGS = ""

DEFAULT_PARALLEL_WORKERS = 1

DEFAULT_ENVIRONMENTS_DIR = os.path.join(os.getcwd(), ".envs")


@contextmanager
def _extra_sys_path(extra_path: str):
    sys.path.insert(0, extra_path)

    yield

    sys.path.remove(extra_path)


def _get_import_path(klass: Type):
    return f"{klass.__module__}.{klass.__name__}"


def _reload_module(import_path: str):
    if not import_path:
        return

    module_path, _, _ = import_path.rpartition(".")
    module = importlib.import_module(module_path)
    importlib.reload(module)


class BaseSettings(_BaseSettings):
    @no_type_check
    def __setattr__(self, name, value):
        """
        Patch __setattr__ to be able to use property setters.
        From:
            https://github.com/pydantic/pydantic/issues/1577#issuecomment-790506164
        """
        try:
            super().__setattr__(name, value)
        except ValueError as e:
            setters = inspect.getmembers(
                self.__class__,
                predicate=lambda x: isinstance(x, property) and x.fset is not None,
            )
            for setter_name, func in setters:
                if setter_name == name:
                    object.__setattr__(self, name, value)
                    break
            else:
                raise e

    def dict(self, by_alias=True, exclude_unset=True, exclude_none=True, **kwargs):
        """
        Ensure that aliases are used, and that unset / none fields are ignored.
        """
        return super().dict(
            by_alias=by_alias,
            exclude_unset=exclude_unset,
            exclude_none=exclude_none,
            **kwargs,
        )

    def json(self, by_alias=True, exclude_unset=True, exclude_none=True, **kwargs):
        """
        Ensure that aliases are used, and that unset / none fields are ignored.
        """
        return super().json(
            by_alias=by_alias,
            exclude_unset=exclude_unset,
            exclude_none=exclude_none,
            **kwargs,
        )


class Settings(BaseSettings):
    class Config:
        env_file = ENV_FILE_SETTINGS
        env_prefix = ENV_PREFIX_SETTINGS

    debug: bool = Field(default=True, description="Enable debug mode")

    parallel_workers: int = Field(
        default=DEFAULT_PARALLEL_WORKERS,
        description=f"Number of workers to run document-generation. "
        f"Default is {DEFAULT_PARALLEL_WORKERS}.",
    )

    parallel_workers_timeout: int = Field(
        default=5,
        description=f"Grace timeout to wait until the workers shut down when stopping DGServer.",
    )

    environments_dir: str = Field(
        default=DEFAULT_ENVIRONMENTS_DIR,
        description=f"Directory used to store custom environments.",
    )

    server_name: str = Field(default="dg-server", description=f"Name of the server.")

    server_version: str = Field(
        default=__version__, description=f"Version of the server."
    )

    host: str = Field(
        default="127.0.0.1", description=f"Host where to listen for connections."
    )

    http_port: int = Field(
        default=8080, description=f"Port where to listen for HTTP / REST connections."
    )

    root_path: str = Field(
        default="",
        description=f"Set the ASGI root_path for applications submounted below a given URL path.",
    )

    # 这里是mongodb的配置
    mongodb_host: str = Field(
        default="localhost",
        description=f"Host where to listen for mongodb connections.",
    )

    mongo_port: int = Field(
        default=27017, description=f"Port where to listen for mongo connections."
    )

    mongo_database: str = Field(
        default="doc_gen", description=f"which database use in mongodb"
    )

    # 以上是fastapi-uvicorn服务器所需配置
    # 以下增加扩展配置参数

    redis_url: str = Field(
        default="redis://124.70.207.36:6379", description=f"Redis url"
    )

    para_gen_stream_name: str = Field(
        default="paragen", description=f"In which stream we handle task"
    )

    para_gen_workers: int = Field(
        default=1, description=f"Number of workers to invoke llm"
    )

    para_gen_groupname: str = Field(
        default="default", description=f"In which stream we handle task"
    )

    minio_url: str = Field(default="124.70.207.36:9900")

    minio_access_key: str = Field(default="BDILAB")

    minio_secret_key: str = Field(default="BDILAB1124")

    minio_secure: bool = Field(default=False)

    default_large_model: str = Field(default="qwen1.5-14b-chat")

    OPENAI_API_KEY: str = Field(default="EMPTY")

    OPENAI_API_BASE: str = Field(default="http://124.70.207.36:7002/v1")

    clickhouse_url: str = Field(
        default="clickhouse://default:123456@124.70.207.36:8123/default",
        description=f"format: clickhouse://user:password@server_host:port/db",
    )

    Text2SQL_JSON_PATH: str = Field(default=r"../text2sql_json/生产建设完成情况周报表问答对.json")

    DEFAULT_KR_TOPIC: str = Field(default="煤矿生产监管检查")

    DEFAULT_EMBEDDING_MODEL: str = Field(default="bce-embedding-base_v1")

    APP_HOST: str = Field(default="0.0.0.0")
    APP_PORT: str = Field(default="6006")

    NEO4J_BOLT: str = Field(default="bolt://localhost:7687")
    NEO4J_USERNAME: str = Field(default="neo4j")
    NEO4J_PASSWORD: str = Field(default="12341234")

    REDIS_HOST: str = Field(default="localhost")

    PROJECT_NAME: str = Field(default="KnowledgeRetriever")