import logging
import os

# Create a logger
logger = logging.getLogger("chatbot_logger")
logger.setLevel(logging.DEBUG)

# Create a logs directory if it doesn't exist
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# File handler to write logs to a file
file_handler = logging.FileHandler(os.path.join(log_dir, "chatbot.log"))
file_handler.setLevel(logging.DEBUG)

# Console handler to output logs to the console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Formatter for log messages
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Example usage
if __name__ == "__main__":
    logger.info("Logger initialized successfully.")
    logger.debug("This is a debug message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.critical("This is a critical message.")