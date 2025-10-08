"""Safety utilities for filtering and validating LLM inputs and outputs."""
from google import genai
from google.genai import types
from src.app.logger_config import setup_logger, log_safety_trigger

# Set up logger for this module
logger = setup_logger(__name__)


def pre_filter_input(text: str) -> tuple[bool, str]:
    """
    Pre-filter to block jailbreak attempts before reaching LLM.

    Args:
        text: User input text to validate

    Returns:
        Tuple of (is_safe, error_message)
        - is_safe: True if input passes filter, False if blocked
        - error_message: Error message to show user if blocked, empty string if safe
    """
    if not text or not text.strip():
        return True, ""

    # Banned words that indicate jailbreak attempts
    banned_words = [
        "ignore", "override", "reinitialize", "bypass",
        "system prompt", "forget", "disregard", "new instructions"
    ]

    text_lower = text.lower()
    for word in banned_words:
        if word in text_lower:
            log_safety_trigger(logger, "pre-filter", f"Banned word detected: '{word}'", text)
            logger.info(f"Blocked input containing: '{word}'")
            return False, "Sorry, I cannot process this request."

    logger.debug("Pre-filter passed")
    return True, ""


def is_fashion_related(client: genai.Client, text: str) -> bool:
    """
    Post-filter to verify LLM response stayed on-topic about fashion.
    Uses a separate LLM call to validate the response content.

    Args:
        client: Gemini client instance
        text: LLM output text to validate

    Returns:
        True if text is fashion-related, False otherwise
    """
    if not text or not text.strip():
        return True

    check_prompt = (
        "Answer strictly YES or NO.\n"
        "Is the following text about fashion, clothing, styling, or wardrobe?\n"
        f"Text: {text}"
    )

    try:
        logger.debug(f"Running post-filter check on text: '{text[:50]}...'")
        result = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[{"text": check_prompt}],
            config=types.GenerateContentConfig(temperature=0.0),
        )
        answer = result.candidates[0].content.parts[0].text.strip().upper()
        is_related = answer == "YES"

        if not is_related:
            log_safety_trigger(logger, "post-filter", "Content not fashion-related", text)
            logger.warning(f"Post-filter blocked non-fashion content")

        logger.debug(f"Post-filter result: {answer}")
        return is_related

    except Exception as e:
        logger.error(f"Post-filter check failed: {type(e).__name__} - {str(e)}")
        # Default to True to avoid blocking legitimate responses on error
        return True


def get_safety_settings():
    """
    Get standard Gemini safety settings for all LLM calls.
    Blocks hate speech, dangerous content, sexually explicit content, and harassment.

    Returns:
        List of SafetySetting objects
    """
    return [
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        ),
    ]


def check_safety_ratings(candidate) -> tuple[bool, str]:
    """
    Monitor safety ratings in streaming responses.

    Args:
        candidate: Gemini response candidate object

    Returns:
        Tuple of (is_safe, error_message)
        - is_safe: True if safe, False if safety filter triggered
        - error_message: Error message to show user if unsafe
    """
    if not candidate.safety_ratings:
        return True, ""

    for rating in candidate.safety_ratings:
        if rating.probability not in ("NEGLIGIBLE", "LOW"):
            log_safety_trigger(
                logger,
                "gemini-safety",
                f"Category: {rating.category}, Probability: {rating.probability}"
            )
            logger.warning(f"Gemini safety filter triggered: {rating.category} ({rating.probability})")
            return False, "Sorry, I cannot answer that due to safety restrictions. Please ask me something about fashion."

    logger.debug("Safety ratings check passed")
    return True, ""
