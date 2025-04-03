from .settings import Settings
from .server import Server
from .utils.logging import get_logger
import asyncio
import logging

logger = get_logger()
async def main():
    settings = Settings()

    logger.info(f"The parameters of the server: \n{settings.__str__()}")
    logger.info(f"""
██████╗ ██████╗ ██╗██╗      █████╗ ██████╗ 
██╔══██╗██╔══██╗██║██║     ██╔══██╗██╔══██╗
██████╔╝██║  ██║██║██║     ███████║██████╔╝
██╔══██╗██║  ██║██║██║     ██╔══██║██╔══██╗
██████╔╝██████╔╝██║███████╗██║  ██║██████╔╝
╚═════╝ ╚═════╝ ╚═╝╚══════╝╚═╝  ╚═╝╚═════╝                                          
""")
    server = Server(settings)
    await server.start()

    logger.info(f"server start successfully")

if __name__=="__main__":
    asyncio.run(main())