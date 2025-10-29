"""
Jina AI classification client.

Provides wrapper for Jina AI text classification API with error handling and logging.
"""

import logging
import time

import requests

logger = logging.getLogger("whatsapp_analyzer")


def classify_texts(
    inputs: str,
    labels: list[str],
    model: str = "jina-embeddings-v3",
    api_key: str | None = None,
    timeout: int = 30,
) -> dict:
    """
    Classify text using Jina AI classification API.

    Args:
        inputs: Input text to classify
        labels: List of possible labels/categories
        model: Jina model name (default: "jina-embeddings-v3")
        api_key: Jina API key (required)
        timeout: Request timeout in seconds

    Returns:
        API response as dictionary

    Raises:
        ValueError: If API key is not provided
        requests.HTTPError: If API request fails

    Example:
        >>> result = classify_texts(
        ...     "I love programming",
        ...     ["technology", "sports", "food"],
        ...     api_key="your_key"
        ... )
        >>> result['data'][0]['prediction']
        'technology'
    """
    start = time.time()
    logger.debug(f"classify_texts: model={model}, labels_count={len(labels) if labels else 0}")

    if api_key is None:
        raise ValueError("Jina API Key must be provided.")

    if not labels:
        raise ValueError("Labels list cannot be empty.")

    url = "https://api.jina.ai/v1/classify"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    data = {
        "model": model,
        "input": inputs,
        "labels": labels,
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=timeout)
        response.raise_for_status()

        elapsed = time.time() - start
        logger.debug(f"classify_texts response: status={response.status_code}, time={elapsed:.2f}s")

        # Log response size for debugging (avoid logging full content)
        text_len = len(response.text or "")
        logger.debug(f"classify_texts response length: {text_len} bytes")

        if logger.isEnabledFor(logging.DEBUG):
            # Only log truncated response in debug mode
            truncated = (response.text[:1000] + "...") if text_len > 1000 else response.text
            logger.debug(f"classify_texts truncated response: {truncated}")

        return response.json()

    except requests.exceptions.Timeout:
        logger.error(f"Jina API request timed out after {timeout}s")
        raise
    except requests.exceptions.HTTPError as e:
        logger.error(f"Jina API HTTP error: {e.response.status_code} - {e.response.text}")
        raise
    except requests.exceptions.RequestException as e:
        logger.exception(f"Jina API request failed: {e}")
        raise
    except Exception as e:
        logger.exception(f"Unexpected error in classify_texts: {e}")
        raise
