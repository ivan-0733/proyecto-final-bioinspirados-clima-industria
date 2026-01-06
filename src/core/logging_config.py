"""
Structured logging configuration using structlog.
"""
import logging
import sys
from pathlib import Path
from typing import Optional
import structlog
from structlog.processors import JSONRenderer, TimeStamper, StackInfoRenderer, format_exc_info
from structlog.stdlib import add_log_level, ProcessorFormatter


def setup_logging(
    log_file: Optional[Path] = None,
    level: str = "INFO",
    json_logs: bool = True
) -> structlog.BoundLogger:
    """
    Configure structured logging for the entire system.
    
    Args:
        log_file: Path to log file (optional, uses stdout if None)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_logs: Whether to use JSON format (True) or human-readable (False)
    
    Returns:
        Configured logger instance
    """
    log_level = getattr(logging, level.upper())
    
    # Shared processors for all loggers
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        add_log_level,
        TimeStamper(fmt="iso", utc=True),
        StackInfoRenderer(),
        format_exc_info,
    ]
    
    # Configure stdlib logging
    logging.basicConfig(
        format="%(message)s",
        level=log_level,
        handlers=[],
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    if json_logs:
        console_handler.setFormatter(
            ProcessorFormatter(
                processor=JSONRenderer(),
                foreign_pre_chain=shared_processors,
            )
        )
    else:
        console_handler.setFormatter(
            ProcessorFormatter(
                processor=structlog.dev.ConsoleRenderer(colors=True),
                foreign_pre_chain=shared_processors,
            )
        )
    
    # File handler (if specified)
    handlers = [console_handler]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setFormatter(
            ProcessorFormatter(
                processor=JSONRenderer(),
                foreign_pre_chain=shared_processors,
            )
        )
        handlers.append(file_handler)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    for handler in handlers:
        root_logger.addHandler(handler)
    root_logger.setLevel(log_level)
    
    # Configure structlog
    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    return structlog.get_logger()


def get_logger(name: str = None) -> structlog.BoundLogger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Configured logger with bound context
    """
    if name:
        return structlog.get_logger(name)
    return structlog.get_logger()


def bind_context(**kwargs) -> None:
    """
    Bind context variables for all subsequent log calls in this thread.
    
    Example:
        bind_context(generation=42, experiment="test")
    """
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_context() -> None:
    """Clear all bound context variables."""
    structlog.contextvars.clear_contextvars()
