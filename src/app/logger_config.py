import logging
import sys
from pathlib import Path
from datetime import datetime

# Create logs directory if it doesn't exist
LOGS_DIR = Path(__file__).parent.parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Log file path with timestamp
LOG_FILE = LOGS_DIR / f"fashion_app_{datetime.now().strftime('%Y%m%d')}.log"


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with both file and console handlers.

    Args:
        name: Logger name (usually __name__ from calling module)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_formatter = logging.Formatter(
        fmt='%(levelname)s - %(name)s - %(message)s'
    )

    # File handler (detailed logs)
    file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)  # Log everything to file
    file_handler.setFormatter(detailed_formatter)

    # Console handler (less verbose)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)  # Only INFO and above to console
    console_handler.setFormatter(console_formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def log_api_call(logger: logging.Logger, api_name: str, endpoint: str, params: dict = None):
    """
    Log API call details.

    Args:
        logger: Logger instance
        api_name: Name of the API (e.g., "Weather API", "Gemini API")
        endpoint: API endpoint being called
        params: Optional parameters being sent
    """
    params_str = f" with params: {params}" if params else ""
    logger.info(f"Calling {api_name} - {endpoint}{params_str}")


def log_api_success(logger: logging.Logger, api_name: str, response_summary: str = None):
    """
    Log successful API response.

    Args:
        logger: Logger instance
        api_name: Name of the API
        response_summary: Optional summary of response
    """
    summary_str = f": {response_summary}" if response_summary else ""
    logger.info(f"{api_name} call successful{summary_str}")


def log_api_error(logger: logging.Logger, api_name: str, error: Exception, retry_count: int = 0):
    """
    Log API error with details.

    Args:
        logger: Logger instance
        api_name: Name of the API
        error: Exception that occurred
        retry_count: Current retry attempt number
    """
    retry_str = f" (attempt {retry_count})" if retry_count > 0 else ""
    logger.error(f"{api_name} call failed{retry_str}: {type(error).__name__} - {str(error)}")


def log_safety_trigger(logger: logging.Logger, filter_type: str, reason: str, input_text: str = None):
    """
    Log safety filter triggers.

    Args:
        logger: Logger instance
        filter_type: Type of filter (pre-filter, post-filter, gemini-safety)
        reason: Reason for trigger
        input_text: Optional sanitized input text (first 50 chars)
    """
    text_preview = f" - Input: '{input_text[:50]}...'" if input_text else ""
    logger.warning(f"Safety filter triggered [{filter_type}]: {reason}{text_preview}")


def log_rag_retrieval(logger: logging.Logger, query: str, num_docs: int, success: bool = True):
    """
    Log RAG document retrieval.

    Args:
        logger: Logger instance
        query: Search query used
        num_docs: Number of documents retrieved
        success: Whether retrieval was successful
    """
    status = "successful" if success else "failed"
    logger.info(f"RAG retrieval {status} - Query: '{query[:50]}...' - Docs: {num_docs}")


# Create a default logger for the app
app_logger = setup_logger("fashion_app")
