"""
Logging Configuration for PolyQuant 2.0

This module sets up structured logging using structlog for better observability.

WHY STRUCTURED LOGGING?
-----------------------
Traditional logs are just strings that are hard to parse and analyze.
Structured logs output JSON, making them:
1. Easy to search and filter in log aggregation tools (e.g., Grafana Loki)
2. Machine-readable for automated alerting
3. Consistent across the entire application

USAGE:
------
    from polyquant.utils.logging_config import get_logger
    
    logger = get_logger(__name__)
    
    # Basic logging
    logger.info("Market found", market_id="123", price=0.65)
    
    # With exception info
    try:
        risky_operation()
    except Exception:
        logger.exception("Operation failed")

LOG LEVELS:
-----------
- DEBUG: Detailed info for debugging (e.g., raw API responses)
- INFO: Normal operations (e.g., "Trade executed", "Market scanned")
- WARNING: Something unexpected but not critical (e.g., "Retry needed")
- ERROR: Something failed (e.g., "API call failed")
- CRITICAL: System is in danger (e.g., "Kill switch activated")
"""

import logging
import sys
from typing import Any

import structlog
from structlog.types import Processor

from polyquant.utils.config import config


def configure_logging() -> None:
    """
    Configure structured logging for the entire application.
    
    This function should be called once at application startup, before
    any logging is performed.
    
    The configuration adapts based on the LOG_JSON environment variable:
    - True: Outputs JSON for production (machine-readable)
    - False: Outputs colored, human-readable format for development
    """
    
    # ==========================================================================
    # Step 1: Configure the shared processors
    # These run on every log message to add context and format
    # ==========================================================================
    
    shared_processors: list[Processor] = [
        # Add log level as a string (e.g., "info", "error")
        structlog.stdlib.add_log_level,
        
        # Add the logger name (usually the module name)
        structlog.stdlib.add_logger_name,
        
        # Add timestamp in ISO format
        structlog.processors.TimeStamper(fmt="iso"),
        
        # If there's an exception, format it nicely
        structlog.processors.format_exc_info,
        
        # Handle Unicode properly
        structlog.processors.UnicodeDecoder(),
    ]
    
    # ==========================================================================
    # Step 2: Choose the appropriate renderer based on config
    # ==========================================================================
    
    if config.log_json:
        # Production: JSON output for log aggregation tools
        # Each log line is a valid JSON object
        renderer: Processor = structlog.processors.JSONRenderer()
    else:
        # Development: Colored, human-readable output
        # Much easier to read in a terminal
        renderer = structlog.dev.ConsoleRenderer(
            colors=True,
            exception_formatter=structlog.dev.plain_traceback,
        )
    
    # ==========================================================================
    # Step 3: Configure structlog
    # ==========================================================================
    
    structlog.configure(
        processors=[
            # Filter out log messages below the configured level
            structlog.stdlib.filter_by_level,
            
            # Add shared processors
            *shared_processors,
            
            # Prepare for the final renderer
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        # Use structlog's logger factory that integrates with stdlib logging
        logger_factory=structlog.stdlib.LoggerFactory(),
        
        # Cache the logger for performance
        cache_logger_on_first_use=True,
    )
    
    # ==========================================================================
    # Step 4: Configure stdlib logging (needed for third-party libraries)
    # ==========================================================================
    
    # Create a formatter that structlog can use
    formatter = structlog.stdlib.ProcessorFormatter(
        # These run ONLY on logs from stdlib loggers (e.g., httpx, openai)
        foreign_pre_chain=shared_processors,
        
        # The final renderer (JSON or Console)
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )
    
    # Set up the root handler to output to stdout
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    
    # Configure the root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(config.log_level)
    
    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def get_logger(name: str) -> Any:
    """
    Get a structured logger for a module.
    
    This is the primary way to get a logger throughout the application.
    The logger automatically includes context like timestamps and log levels.
    
    Args:
        name: Usually __name__ from the calling module
        
    Returns:
        A structlog logger instance
        
    Example:
        logger = get_logger(__name__)
        logger.info("Market scanned", market_count=42)
    """
    return structlog.get_logger(name)


# Auto-configure logging when this module is imported for the first time
# This ensures consistent logging behavior across all imports
configure_logging()
