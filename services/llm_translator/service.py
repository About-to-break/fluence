from logging_tools import logging_tools
from config import load_config

def serve():
    config = load_config()
    logger = logging_tools.get_logger(
        level=config.LOG_LEVEL,
        file=config.LOG_FILE
    )
    logger.info("Starting server")

    logger.info("Performing basic environment check...")


    logger.info("Init server DONE")

if __name__ == '__main__':
    serve()