import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logging():
    """
    Configures the logging system for the application.
    Logs are written to 'logs/app.log' and the console.
    """
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, "app.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            RotatingFileHandler(
                log_file, maxBytes=10*1024*1024, backupCount=5  # 10MB per file, keep 5 backups
            ),
            logging.StreamHandler()
        ]
    )
    
    # Set level for specific loggers if needed
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    
    logger = logging.getLogger(__name__)
    logger.info("Logging initialized. Logs are written to %s", log_file)
