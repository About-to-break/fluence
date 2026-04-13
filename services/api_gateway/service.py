import logging

import uvicorn
from logging_tools import configure_global_logging

from config import load_config
from internal.app import create_app


def serve():
    config = load_config()

    configure_global_logging(
        level=config.LOG_LEVEL,
        file=config.LOG_FILE,
    )

    logging.info("Starting API gateway")

    app = create_app(config)

    uvicorn.run(
        app,
        host=config.API_HOST,
        port=config.API_PORT,
        log_level=logging.getLevelName(config.LOG_LEVEL).lower(),
    )


if __name__ == "__main__":
    serve()
