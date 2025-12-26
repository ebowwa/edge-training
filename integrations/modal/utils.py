"""Modal utility functions for reliability."""

import time
import logging
from functools import wraps
from typing import Any, Callable, TypeVar, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

T = TypeVar('T')


def with_retry(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
) -> Callable:
    """
    Decorator for automatic retry with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch and retry
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}): {e}"
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"{func.__name__} failed after {max_retries + 1} attempts")
            
            raise last_exception
        return wrapper
    return decorator


def chunked_upload(
    source_path: Path,
    upload_fn: Callable[[Path], Any],
    chunk_size_mb: int = 100,
    max_retries: int = 3
) -> bool:
    """
    Upload large files in chunks with retry logic.
    
    Args:
        source_path: Path to file or directory to upload
        upload_fn: Function to call for upload (takes path, returns success)
        chunk_size_mb: Max size per chunk in MB (for splitting large dirs)
        max_retries: Retries per chunk
        
    Returns:
        True if successful
    """
    if not source_path.exists():
        raise FileNotFoundError(f"Source not found: {source_path}")
    
    # If it's a file, just upload with retry
    if source_path.is_file():
        retry_upload = with_retry(max_retries=max_retries)(upload_fn)
        retry_upload(source_path)
        return True
    
    # If it's a directory, upload in chunks
    logger.info(f"Chunked upload from {source_path}")
    
    # Group files by split to upload separately
    splits = [d for d in source_path.iterdir() if d.is_dir()]
    
    for split_dir in splits:
        logger.info(f"Uploading {split_dir.name}...")
        retry_upload = with_retry(max_retries=max_retries)(upload_fn)
        
        try:
            retry_upload(split_dir)
            logger.info(f"✓ {split_dir.name} uploaded")
        except Exception as e:
            logger.error(f"✗ {split_dir.name} failed: {e}")
            raise
    
    return True


def estimate_upload_time(size_bytes: int, bandwidth_mbps: int = 10) -> float:
    """
    Estimate upload time given size and bandwidth.
    
    Args:
        size_bytes: File/directory size in bytes
        bandwidth_mbps: Upload bandwidth in Mbps
        
    Returns:
        Estimated time in seconds
    """
    size_mb = size_bytes / (1024 ** 2)
    return (size_mb * 8) / bandwidth_mbps


def get_directory_size(path: Path) -> int:
    """Get total size of directory in bytes."""
    total = 0
    for item in path.rglob('*'):
        if item.is_file():
            total += item.stat().st_size
    return total
