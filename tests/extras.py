from fastapi import Request
import os
import time
import hmac
from hashlib import sha256
from strukt.logging import StruktLogger
from strukt.prompts import DEFAULT_CLASSIFIER_TEMPLATE

logger = StruktLogger("extras")


# Helper function for HMAC verification
def _verify_elevenlabs_signature(request: Request, payload: str) -> bool:
    """Verify ElevenLabs HMAC signature"""
    try:
        # Get the signature header
        signature_header = request.headers.get("elevenlabs-signature")
        if not signature_header:
            logger.error("Missing ElevenLabs-Signature header")
            return False

        # Parse timestamp and signature
        try:
            timestamp = signature_header.split(",")[0][2:]  # Remove "t="
            hmac_signature = signature_header.split(",")[1]  # Get "v0=hash"
        except IndexError:
            logger.error("Invalid signature header format")
            return False

        # Validate timestamp (within 30 minutes)
        current_time = int(time.time())
        tolerance = current_time - (30 * 60)  # 30 minutes ago
        if int(timestamp) < tolerance:
            logger.error(
                f"Webhook timestamp {timestamp} is too old (current: {current_time})"
            )
            return False

        # Get the secret from environment
        secret = os.getenv("HMAC_SECRET_11LABS")
        if not secret:
            logger.error("HMAC_SECRET_11LABS not configured")
            return False

        # Create the expected signature
        full_payload_to_sign = f"{timestamp}.{payload}"
        mac = hmac.new(
            key=secret.encode("utf-8"),
            msg=full_payload_to_sign.encode("utf-8"),
            digestmod=sha256,
        )
        expected_signature = "v0=" + mac.hexdigest()

        # Compare signatures
        if hmac_signature != expected_signature:
            logger.error(
                f"HMAC signature mismatch. Expected: {expected_signature}, Got: {hmac_signature}"
            )
            return False

        logger.info("ElevenLabs HMAC signature verified successfully")
        return True

    except Exception as e:
        logger.error(f"Error verifying HMAC signature: {e}")
        return False

DEFAULT_CLASSIFIER_TEMPLATE = """
Current UTC: {current_time}

You are an intent classifier.

Goal:
- Decide if the user input contains one or several distinct, actionable requests
- Extract minimal, non-overlapping spans for each request
- Assign the best-fitting type to each span

Constraints:
- Use only types from {allowed_types}. If none fits, use "general".
- Return at most {max_parts} parts.
- Output arrays must have equal length and aligned indices.
- Confidence values must be within [0.0, 1.0].
- Do not include any prose or explanation or other text in your response.

Policies:
- Prefer minimal parts; avoid creating multiple parts for closely related statements about the same fact or the same request. (e.g. "I live in Beirut" and "What's the time in Beirut?" should be one part)
- Discard parts that are not actionable or not relevant to the user's request. (e.g. "I live in Dubai" is not actionable, so it should be discarded)
- Parts that may lead to useful memory extraction should be kept and classified as "memory_extraction". (e.g. "I like Pizza" should be kept)
- Do not over-fragment tightly coupled instructions aimed at a single target.
- Prefer spans that can be executed or answered directly.
- If prior user preferences reasonably disambiguate a vague request, refine the span text accordingly.
- If the user asks about their own stored information, preferences, routines, or habits, classify as "no-op-memory-response".

Output specification (must be valid JSON):
- query_types: array of strings
- confidences: array of numbers between 0 and 1
- parts: array of strings
- The three arrays must have the same length and aligned indices.

User request:
{text}
"""