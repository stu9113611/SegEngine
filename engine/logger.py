import logging
from rich.logging import RichHandler


class Logger:
    def __init__(self) -> None:
        FORMAT = "%(message)s"
        logging.basicConfig(
            level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
        )
        self.logger = logging.getLogger("rich")

    def debug(self, title: str, msg: str) -> None:
        self.logger.debug(f"[{title}] {msg}")

    def info(self, title: str, msg: str) -> None:
        self.logger.info(f"[{title}] {msg}")

    def warn(self, title: str, msg: str) -> None:
        self.logger.warn(f"[{title}] {msg}")

    def error(self, title: str, msg: str) -> None:
        self.logger.error(f"[{title}] {msg}")
