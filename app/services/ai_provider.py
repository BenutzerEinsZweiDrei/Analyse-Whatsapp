"""
g4f (GPT4Free) client wrapper.

Provides wrapper around g4f library for AI text generation with error handling.
Enhanced with structured JSON prompts and validation.
"""

import json
import logging
import re

import g4f

logger = logging.getLogger("whatsapp_analyzer")


# Structured JSON schema for AI responses
PERSONALITY_SCHEMA = """
{
  "schema_version": "1.0",
  "traits": {
    "openness": {
      "score": <float 0.0-1.0>,
      "label": "<low|medium|high>",
      "confidence": <float 0.0-1.0>,
      "evidence": [{"message_index": <int>, "author": "<string>", "snippet": "<string>"}]
    },
    "conscientiousness": { ... },
    "extraversion": { ... },
    "agreeableness": { ... },
    "neuroticism": { ... }
  },
  "summary_text": "<concise summary paragraph, max 200 words>",
  "warnings": ["<reasons for low confidence or missing data>"]
}
"""

# Few-shot examples for better JSON generation
FEW_SHOT_EXAMPLES = """
Example 1 - High activity user:
Input: {"positive_topics": ["python", "programming"], "emotion_variability": 1.2, "keywords": ["code", "learn"]}
Output:
{
  "schema_version": "1.0",
  "traits": {
    "openness": {"score": 0.75, "label": "high", "confidence": 0.8, "evidence": [{"message_index": 1, "author": "user", "snippet": "I love learning new things"}]},
    "conscientiousness": {"score": 0.65, "label": "medium", "confidence": 0.7, "evidence": []},
    "extraversion": {"score": 0.55, "label": "medium", "confidence": 0.6, "evidence": []},
    "agreeableness": {"score": 0.70, "label": "high", "confidence": 0.65, "evidence": []},
    "neuroticism": {"score": 0.40, "label": "low", "confidence": 0.75, "evidence": []}
  },
  "summary_text": "The user shows high openness to experience with strong interest in learning.",
  "warnings": []
}
"""


def generate_profile(
    analysis_data: dict, model: str = "gpt-4o-mini", return_json: bool = False
) -> str:
    """
    Generate psychological profile using g4f AI with structured output.

    Tries Client API first, falls back to ChatCompletion.create for compatibility.
    Enhanced with structured prompts for better, verifiable JSON output.

    Args:
        analysis_data: Analysis dictionary to send to AI
        model: Model name to use (default: "gpt-4o-mini")
        return_json: If True, attempt to return parsed JSON; otherwise return text (default: False)

    Returns:
        Generated profile text (or JSON dict if return_json=True and parsing succeeds)

    Raises:
        Various g4f.errors exceptions for different failure modes

    Example:
        >>> profile = generate_profile({"sentiment": "positive", ...})
        >>> "psychological" in profile.lower() or "openness" in profile.lower()
        True
    """
    logger.debug(f"Generating AI profile with model={model}, return_json={return_json}")

    # Build enhanced structured prompt
    analysis_json = json.dumps(analysis_data, ensure_ascii=False, indent=2)

    # Create structured prompt with schema and examples
    message = f"""You are a psychological profiling expert. Based on the WhatsApp conversation analysis below, create a structured personality profile.

IMPORTANT: Respond with VALID JSON ONLY, following this exact schema:

{PERSONALITY_SCHEMA}

Guidelines:
1. Use the Big Five personality traits (OCEAN)
2. Score each trait from 0.0 (very low) to 1.0 (very high)
3. Label as "low" (<0.4), "medium" (0.4-0.7), or "high" (>0.7)
4. Provide confidence scores (0.0-1.0) based on evidence strength
5. Include evidence snippets when available
6. Keep summary_text under 200 words
7. Add warnings for any uncertainties or missing data
8. Output ONLY valid JSON, no additional text

Here's an example output format:
{FEW_SHOT_EXAMPLES}

Analysis data:
{analysis_json}

Now generate the personality profile in valid JSON format:"""

    response = None

    # Try using the Client API first (recommended for newer g4f versions)
    try:
        client = g4f.Client()
        response_obj = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": message}],
        )

        # Extract content from response
        if hasattr(response_obj, "choices") and response_obj.choices:
            response = response_obj.choices[0].message.content
        elif hasattr(response_obj, "content"):
            response = response_obj.content
        else:
            response_type = type(response_obj).__name__
            response_attrs = [a for a in dir(response_obj) if not a.startswith("_")][:5]
            logger.warning(
                f"Unexpected response structure from Client API: type={response_type}, attrs={response_attrs}"
            )
            raise AttributeError(f"Unable to extract content from response (type: {response_type})")

        logger.debug("g4f Client API succeeded")

    except (AttributeError, ImportError, KeyError) as client_error:
        # Fallback to old API for compatibility
        logger.debug(f"Client API failed: {client_error}, trying ChatCompletion.create")

        try:
            response = g4f.ChatCompletion.create(
                model=g4f.models.gpt_4,
                messages=[{"role": "user", "content": message}],
            )
            logger.debug("g4f ChatCompletion.create succeeded")

            if not response:
                raise ValueError("No response content generated from g4f")

        except Exception as fallback_error:
            logger.exception(f"g4f ChatCompletion.create also failed: {fallback_error}")
            raise

    # Process response
    if not response:
        raise ValueError("Empty response from AI service")

    # Try to extract and validate JSON if requested or if response looks like JSON
    if return_json or response.strip().startswith("{"):
        parsed_json = _extract_and_validate_json(response)
        if parsed_json and return_json:
            return parsed_json
        elif not parsed_json:
            logger.warning("Failed to parse JSON from AI response, returning text")

    return response


def _extract_and_validate_json(text: str) -> dict | None:
    """
    Extract and validate JSON from AI response text.

    Handles common issues like markdown code blocks, extra text, etc.

    Args:
        text: Raw response text from AI

    Returns:
        Parsed JSON dict if valid, None otherwise
    """
    # Try to extract JSON from markdown code blocks
    json_pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
    match = re.search(json_pattern, text, re.DOTALL)

    if match:
        json_text = match.group(1)
    else:
        # Try to find JSON object directly
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            json_text = match.group(0)
        else:
            json_text = text

    # Try to parse JSON
    try:
        parsed = json.loads(json_text)

        # Validate schema
        if not isinstance(parsed, dict):
            logger.warning("JSON is not a dict")
            return None

        # Check for required fields
        if "traits" not in parsed:
            logger.warning("JSON missing 'traits' field")
            return None

        # Basic validation of traits
        traits = parsed.get("traits", {})
        required_traits = [
            "openness",
            "conscientiousness",
            "extraversion",
            "agreeableness",
            "neuroticism",
        ]

        for trait in required_traits:
            if trait not in traits:
                logger.warning(f"Missing trait: {trait}")
                return None

            trait_data = traits[trait]
            if not isinstance(trait_data, dict):
                logger.warning(f"Trait {trait} is not a dict")
                return None

            # Check for required fields
            if "score" not in trait_data or "label" not in trait_data:
                logger.warning(f"Trait {trait} missing score or label")
                return None

        logger.info("Successfully validated JSON personality profile")
        return parsed

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON: {e}")
        return None
    except Exception as e:
        logger.warning(f"JSON validation failed: {e}")
        return None


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
