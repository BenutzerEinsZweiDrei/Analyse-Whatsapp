"""
g4f (GPT4Free) client wrapper.

Provides wrapper around g4f library for AI text generation with error handling.
"""

import logging
from typing import Optional, Dict
import g4f

logger = logging.getLogger("whatsapp_analyzer")


def generate_profile(analysis_data: Dict, model: str = "gpt-4o-mini") -> str:
    """
    Generate psychological profile using g4f AI.
    
    Tries Client API first, falls back to ChatCompletion.create for compatibility.
    
    Args:
        analysis_data: Analysis dictionary to send to AI
        model: Model name to use (default: "gpt-4o-mini")
        
    Returns:
        Generated profile text
        
    Raises:
        Various g4f.errors exceptions for different failure modes
        
    Example:
        >>> profile = generate_profile({"sentiment": "positive", ...})
        >>> "psychological" in profile.lower()
        True
    """
    logger.debug(f"Generating AI profile with model={model}")
    
    # Build prompt
    import json
    message = "Erstelle ein kurzes psychologisches Profil anhand der folgenden Whatsapp Analyse \n\n"
    message = f'{message}\n{json.dumps(analysis_data, ensure_ascii=False, indent=2)}'
    
    response = None
    
    # Try using the Client API first (recommended for newer g4f versions)
    try:
        client = g4f.Client()
        response_obj = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": message}],
        )
        
        # Extract content from response
        if hasattr(response_obj, 'choices') and response_obj.choices:
            response = response_obj.choices[0].message.content
        elif hasattr(response_obj, 'content'):
            response = response_obj.content
        else:
            response_type = type(response_obj).__name__
            response_attrs = [a for a in dir(response_obj) if not a.startswith('_')][:5]
            logger.warning(
                f"Unexpected response structure from Client API: type={response_type}, attrs={response_attrs}"
            )
            raise AttributeError(
                f"Unable to extract content from response (type: {response_type})"
            )
        
        logger.debug("g4f Client API succeeded")
        return response
        
    except (AttributeError, ImportError, KeyError) as client_error:
        # Fallback to old API for compatibility
        logger.debug(f"Client API failed: {client_error}, trying ChatCompletion.create")
        
        try:
            response = g4f.ChatCompletion.create(
                model=g4f.models.gpt_4,
                messages=[{"role": "user", "content": message}],
            )
            logger.debug("g4f ChatCompletion.create succeeded")
            
            if response:
                return response
            else:
                raise ValueError("No response content generated from g4f")
                
        except Exception as fallback_error:
            logger.exception(f"g4f ChatCompletion.create also failed: {fallback_error}")
            raise


def handle_g4f_error(error: Exception) -> str:
    """
    Convert g4f exceptions to user-friendly error messages.
    
    Args:
        error: Exception from g4f
        
    Returns:
        User-friendly error message
    """
    error_name = type(error).__name__
    
    # Map known g4f errors to messages
    error_messages = {
        "MissingAuthError": "Authentication required. Please configure AI service credentials.",
        "NoValidHarFileError": "AI service configuration error. HAR file required.",
        "PaymentRequiredError": "AI service requires payment or subscription.",
        "RateLimitError": "AI service rate limit reached. Please try again later.",
        "ConversationLimitError": "AI conversation limit reached. Please try again later.",
        "ProviderNotWorkingError": "AI service provider is currently unavailable.",
        "RetryNoProviderError": "No AI service providers available. Please try again later.",
        "ModelNotFoundError": "Requested AI model not found or unavailable.",
        "TimeoutError": "AI service request timed out. Please try again.",
    }
    
    message = error_messages.get(error_name, f"AI service error: {str(error)}")
    logger.error(f"g4f error ({error_name}): {message}")
    
    return message
