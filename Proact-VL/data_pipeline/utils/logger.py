import logging


class ColorFormatter(logging.Formatter):
    RESET = "\033[0m"
    LEVEL_TO_COLOR = {
        logging.DEBUG: "\033[34m",    # Blue
        logging.INFO: "\033[32m",     # Green
        logging.WARNING: "\033[33m",  # Yellow
        logging.ERROR: "\033[31m",    # Red
        logging.CRITICAL: "\033[35m", # Magenta/Purple
    }
    def __init__(self, fmt: str, msg_color: bool = True) -> None:
        super().__init__(fmt)
        self.msg_color = msg_color

    def format(self, record: logging.LogRecord) -> str:
        original_levelname = record.levelname
        original_msg = record.msg
        original_args = record.args
        color = self.LEVEL_TO_COLOR.get(record.levelno, "")
        formatted_msg = record.getMessage()

        if color:
            record.levelname = f"{color}{original_levelname}{self.RESET}"
            if self.msg_color:
                record.msg = f"{color}{formatted_msg}{self.RESET}"
            record.args = None
        try:
            return super().format(record)
        finally:
            record.levelname = original_levelname
            record.msg = original_msg
            record.args = original_args


class Logger:
    def __init__(
        self, name: str, 
        level: int = logging.INFO,
        fmt: str = '%(asctime)s - %(levelname)s\n%(message)s', 
        msg_color: bool = True
    ) -> None:
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False
        self.set_formatter(
            fmt=fmt,
            msg_color=msg_color
        )

    def set_formatter(self, fmt: str, msg_color: bool = True) -> None:
        formatter = ColorFormatter(fmt, msg_color)
        for handler in self.logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setFormatter(formatter)
                return

        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def get_logger(self) -> logging.Logger:
        return self.logger


if __name__ == "__main__":
    logger = Logger(__name__)
    logger.get_logger().info("Hello, World!")
    logger.get_logger().warning("Warning, World!")
    logger.get_logger().error("Error, World!")
    logger.get_logger().critical("Critical, World!")
    logger.get_logger().debug("Debug, World!")
