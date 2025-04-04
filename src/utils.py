from src.logger import logger
import uuid

logger.info("This is an info message.")

def generate_session_id():
    """
    Generate a unique session ID using UUID4.
    """
    return str(uuid.uuid4())

