import logging
import sys
import os
import threading
from typing import Optional

_lock = threading.Lock()
_default_handler: Optional[logging.Handler] = None

LoggerName = "dg-serve"

from logging import (
    CRITICAL,  # NOQA
    DEBUG,  # NOQA
    ERROR,  # NOQA
    FATAL,  # NOQA
    INFO,  # NOQA
    NOTSET,  # NOQA
    WARN,  # NOQA
    WARNING,  # NOQA
)

log_levels = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

_default_log_level = logging.WARNING


def _get_default_logging_level():
    """
    If TRANSFORMERS_VERBOSITY env var is set to one of the valid choices return that as the new default level. If it is
    not - fall back to `_default_log_level`
    """
    env_level_str = os.getenv("TRANSFORMERS_VERBOSITY", None)
    if env_level_str:
        if env_level_str in log_levels:
            return log_levels[env_level_str]
        else:
            logging.getLogger().warning(
                f"Unknown option TRANSFORMERS_VERBOSITY={env_level_str}, "
                f"has to be one of: {', '.join(log_levels.keys())}"
            )
    return _default_log_level


def _get_library_name() -> str:
    return __name__.split(".")[0]


def _get_library_root_logger() -> logging.Logger:
    return logging.getLogger(_get_library_name())


def _configure_library_root_logger() -> None:
    global _default_handler

    with _lock:
        if _default_handler:
            # This library has already configured the library root logger.
            return
        _default_handler = logging.StreamHandler()  # Set sys.stderr as stream.
        _default_handler.flush = sys.stderr.flush

        # Apply our default configuration to the library root logger.
        library_root_logger = _get_library_root_logger()
        library_root_logger.addHandler(_default_handler)
        library_root_logger.setLevel(_get_default_logging_level())
        library_root_logger.propagate = False


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Return a logger with the specified name.

    This function is not supposed to be directly accessed unless you are writing a custom transformers module.
    """

    if name is None:
        name = _get_library_name()

    _configure_library_root_logger()
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger


import logging


def init_logger(logger, level=_default_log_level, stream=True, file=None,
                format_str='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                stream_level=None, file_level=None):
    """
    初始化并返回一个 logger 对象。

    :param logger_name: logger 的名称。
    :param level: 默认日志级别。
    :param stream: 是否添加 StreamHandler。
    :param file: 日志文件的路径，如果不为 None，则添加 FileHandler。
    :param format_str: 日志的格式字符串。
    :param stream_level: StreamHandler 的日志级别，如果为 None，则使用 level。
    :param file_level: FileHandler 的日志级别，如果为 None，则使用 level。
    :return: 配置后的 logger 对象。
    """
    logger.setLevel(level)
    formatter = logging.Formatter(format_str)

    if stream:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(stream_level if stream_level is not None else level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if file:
        file_handler = logging.FileHandler(file)
        file_handler.setLevel(file_level if file_level is not None else level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
