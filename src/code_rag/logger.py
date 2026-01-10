"""Central logging configuration for the code-rag tool."""

import logging

from .config.config import Config

# Get log level from config
config = Config()
log_level = config.get_log_level()

# Configure logging
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
