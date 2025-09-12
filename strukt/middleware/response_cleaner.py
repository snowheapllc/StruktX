"""Roomi extension middleware for response cleaning and normalization."""

from __future__ import annotations

import re
import inflect
from functools import lru_cache

from strukt.middleware import Middleware
from strukt.types import HandlerResult, InvocationState
from strukt.logging import get_logger

# Initialize the inflect engine for number-to-word conversion
p = inflect.engine()

# Global unit mapping dictionary for use across functions
UNIT_MAPPING = {
    "km": "kilometers",
    "km/h": "kilometers per hour",
    "m": "meters",
    "cm": "centimeters",
    "mm": "millimeters",
    "g": "grams",
    "kg": "kilograms",
    "ml": "milliliters",
    "l": "liters",
    "mph": "miles per hour",
    "ft": "feet",
    "in": "inches",
    "yd": "yards",
    "lb": "pounds",
    "oz": "ounces",
    "min": "minutes",
    "sec": "seconds",
    "hr": "hours",
    "kW": "kilowatts",
    "W": "watts",
    "kWh": "kilowatt hours",
    "GB": "gigabytes",
    "MB": "megabytes",
    "KB": "kilobytes",
    "TB": "terabytes",
    "bps": "bits per second",
    "kbps": "kilobits per second",
    "Mbps": "megabits per second",
    "Gbps": "gigabits per second",
    "Hz": "hertz",
    "kHz": "kilohertz",
    "MHz": "megahertz",
    "GHz": "gigahertz",
}


class ResponseCleanerMiddleware(Middleware):
    """Middleware for cleaning and normalizing responses for better TTS pronunciation."""

    def __init__(self):
        self._log = get_logger(__name__)

    def after_handle(
        self,
        state: InvocationState,
        query_type: str,
        result: HandlerResult,
    ) -> HandlerResult:
        """Clean and normalize the response text for better TTS pronunciation."""
        if result.response:
            try:
                # Handle different response types appropriately
                if isinstance(result.response, dict):
                    # For dict responses, clean only the message field if it exists
                    response_dict = result.response.copy()
                    if "message" in response_dict and isinstance(
                        response_dict["message"], str
                    ):
                        original_message = response_dict["message"]
                        cleaned_message = self._clean_response(original_message)
                        if cleaned_message != original_message:
                            response_dict["message"] = cleaned_message
                            self._log.debug(
                                f"Cleaned message field for {query_type}: {original_message[:50]}... -> {cleaned_message[:50]}..."
                            )
                            return HandlerResult(
                                response=response_dict, status=result.status
                            )
                    # Return original dict if no message field or no cleaning needed
                    return result
                elif isinstance(result.response, str):
                    # For string responses, clean the entire response
                    cleaned_response = self._clean_response(result.response)
                    if cleaned_response != result.response:
                        self._log.debug(
                            f"Cleaned response for {query_type}: {result.response[:50]}... -> {cleaned_response[:50]}..."
                        )
                        return HandlerResult(
                            response=cleaned_response, status=result.status
                        )
                else:
                    # For other types, log a warning and return as-is
                    self._log.warn(
                        f"Response type {type(result.response).__name__} not handled by response cleaner for {query_type}"
                    )
                    return result
            except Exception as e:
                self._log.warn(f"Error cleaning response: {e}")

        return result

    # ---------------------------------------------------------------------------
    # Number helpers
    # ---------------------------------------------------------------------------

    @lru_cache(maxsize=128)
    def _number_to_words(self, number):
        """Convert a number to words with caching for performance"""
        try:
            if isinstance(number, str) and "." in number:
                # Handle decimal numbers
                integer_part, decimal_part = number.split(".")
                integer_words = p.number_to_words(int(integer_part))
                decimal_words = " ".join(
                    [
                        "zero" if d == "0" else p.number_to_words(int(d))
                        for d in decimal_part
                    ]
                )
                return f"{integer_words} point {decimal_words}"
            else:
                # Handle integers
                return p.number_to_words(int(number))
        except (ValueError, TypeError):
            return str(number)

    @lru_cache(maxsize=1024)
    def _normalize_numbers(self, text):
        """
        Convert numeric values to their word form for better TTS pronunciation.
        Handles:
        - Decimal numbers (e.g., "33.02" -> "thirty-three point zero two")
        - Whole numbers (e.g., "40" -> "forty")
        - Percentages (e.g., "40%" -> "forty percent")
        """

        # Handle decimal numbers (like 33.02)
        def decimal_replacer(match):
            full_num = match.group(0)
            return self._number_to_words(full_num)

        # Find standalone decimal numbers (not part of words)
        text = re.sub(r"\b\d+\.\d+\b", decimal_replacer, text)

        # Handle standalone whole numbers (not part of words like dates, versions, etc.)
        # This regex tries to match numbers that aren't part of common patterns like dates or versions
        text = re.sub(
            r"(?<!\d\.)(?<!\d-)(?<![A-Za-z]-)\b(\d+)\b(?![\.\-\/])",
            lambda m: self._number_to_words(m.group(1)),
            text,
        )

        # Handle percentages
        text = re.sub(
            r"(\d+)%", lambda m: f"{self._number_to_words(m.group(1))} percent", text
        )

        # Handle ranges (e.g., "20-30" -> "twenty to thirty")
        text = re.sub(
            r"\b(\d+)-(\d+)\b",
            lambda m: f"{self._number_to_words(m.group(1))} to {self._number_to_words(m.group(2))}",
            text,
        )

        return text

    @lru_cache(maxsize=128)
    def _normalize_units(self, text):
        """
        Convert unit symbols to their spoken form.
        Handles:
        - Temperature units (e.g., "°C" -> "degrees Celsius")
        - Common unit abbreviations
        """
        # Temperature units
        text = re.sub(
            r"(\d+(?:\.\d+)?)°C",
            lambda m: f"{self._normalize_numbers(m.group(1))} degrees Celsius",
            text,
        )
        text = re.sub(
            r"(\d+(?:\.\d+)?)°F",
            lambda m: f"{self._normalize_numbers(m.group(1))} degrees Fahrenheit",
            text,
        )
        text = re.sub(
            r"(\d+(?:\.\d+)?)°",
            lambda m: f"{self._normalize_numbers(m.group(1))} degrees",
            text,
        )
        text = text.replace("%", " percent")

        # Process unit abbreviations using the global mapping
        for abbr, full in UNIT_MAPPING.items():
            # Match unit patterns with a number
            text = re.sub(
                f"(\\d+(?:\\.\\d+)?)\\s*{abbr}\\b",
                lambda m: f"{self._normalize_numbers(m.group(1))} {full}",
                text,
            )

        return text

    @lru_cache(maxsize=128)
    def _normalize_phone_numbers(self, text):
        """
        Convert phone numbers to their spoken form.
        Examples:
        - "123-456-7890" -> "one two three, four five six, seven eight nine zero"
        - "+1 (123) 456-7890" -> "plus one, one two three, four five six, seven eight nine zero"
        """

        # Helper function to pronounce digits individually
        def pronounce_digits(digits):
            """Convert each digit to words and join with spaces"""
            return " ".join(
                [
                    (
                        self._number_to_words(int(d))
                        if d.isdigit()
                        else "plus"
                        if d == "+"
                        else d
                    )
                    for d in digits
                ]
            )

        # North American and international phone number formats
        patterns = [
            # Standard formats
            (
                r"(\d{3})-(\d{3})-(\d{4})",
                lambda m: f"{pronounce_digits(m.group(1))}, {pronounce_digits(m.group(2))}, {pronounce_digits(m.group(3))}",
            ),
            (
                r"\+(\d+)\s*\((\d{3})\)\s*(\d{3})-(\d{4})",
                lambda m: f"plus {pronounce_digits(m.group(1))}, {pronounce_digits(m.group(2))}, {pronounce_digits(m.group(3))}, {pronounce_digits(m.group(4))}",
            ),
            (
                r"\((\d{3})\)\s*(\d{3})-(\d{4})",
                lambda m: f"{pronounce_digits(m.group(1))}, {pronounce_digits(m.group(2))}, {pronounce_digits(m.group(3))}",
            ),
            # International formats
            (
                r"\+(\d{1,3})\s*(\d{2,5})\s*(\d{3,8})",
                lambda m: f"plus {pronounce_digits(m.group(1))}, {pronounce_digits(m.group(2))}, {pronounce_digits(m.group(3))}",
            ),
            (
                r"\+(\d{1,3})\s*(\d{2,3})\s*(\d{2,4})\s*(\d{2,4})",
                lambda m: f"plus {pronounce_digits(m.group(1))}, {pronounce_digits(m.group(2))}, {pronounce_digits(m.group(3))}, {pronounce_digits(m.group(4))}",
            ),
            # Plain sequences of digits (likely phone numbers)
            (
                r"\b(\d{5,7})(\d{5,7})\b",
                lambda m: f"{pronounce_digits(m.group(1))}, {pronounce_digits(m.group(2))}",
            ),
            (
                r"\b(\d{10,15})\b",
                lambda m: ", ".join(
                    [
                        pronounce_digits(m.group(1)[i : i + 3])
                        for i in range(0, len(m.group(1)), 3)
                    ]
                ),
            ),
            # Numbers with country codes
            (
                r"\+(\d{1,3})(\d{9,12})",
                lambda m: f"plus {pronounce_digits(m.group(1))}, {', '.join([pronounce_digits(m.group(2)[i : i + 3]) for i in range(0, len(m.group(2)), 3)])}",
            ),
            (
                r"00(\d{1,3})(\d{9,12})",
                lambda m: f"zero zero {pronounce_digits(m.group(1))}, {', '.join([pronounce_digits(m.group(2)[i : i + 3]) for i in range(0, len(m.group(2)), 3)])}",
            ),
        ]

        # Apply all patterns
        for pattern, replacement in patterns:
            text = re.sub(pattern, replacement, text)

        return text

    @lru_cache(maxsize=128)
    def _normalize_time(self, text):
        """
        Convert time formats to their spoken form.
        Handles 12-hour and 24-hour formats.
        """

        # Regular 12-hour clock time (e.g., "10:30 AM" -> "ten thirty AM")
        def time_replacer(match):
            hours, minutes, ampm = match.groups()
            hours_word = self._number_to_words(int(hours))

            # Handle minutes
            if minutes == "00":
                # If minutes are "00", omit them entirely for a cleaner spoken form like "ten AM"
                return f"{hours_word} {ampm.upper()}"  # Ensure AM/PM is uppercase for consistency
            else:
                minutes_val = int(minutes)
                # If minutes start with 0 (e.g., "05") and are single digit, say "oh five"
                # Otherwise, convert the full minute number to words.
                if minutes.startswith("0") and len(minutes) > 1 and minutes_val < 10:
                    minutes_word = f"oh {self._number_to_words(minutes_val)}"
                else:
                    minutes_word = self._number_to_words(minutes_val)
                return f"{hours_word} {minutes_word} {ampm.upper()}"  # Ensure AM/PM is uppercase

        # Match time patterns
        text = re.sub(r"(\d{1,2}):(\d{2})\s*(AM|PM|am|pm)", time_replacer, text)

        # 24-hour clock time (e.g., "14:30" -> "fourteen thirty")
        def military_time_replacer(match):
            hours, minutes = match.groups()
            hours_word = self._number_to_words(int(hours))

            if minutes == "00":
                # For military time xx:00, say "xx hundred hours"
                return f"{hours_word} hundred hours"
            else:
                minutes_val = int(minutes)
                # If minutes start with 0 (e.g., "05") and are single digit, say "oh five"
                # Otherwise, convert the full minute number to words.
                if minutes.startswith("0") and len(minutes) > 1 and minutes_val < 10:
                    minutes_word = f"oh {self._number_to_words(minutes_val)}"
                else:
                    minutes_word = self._number_to_words(minutes_val)
                return f"{hours_word} {minutes_word} hours"

        # Match 24-hour clock time (when not followed by AM/PM)
        text = re.sub(
            r"(\d{1,2}):(\d{2})(?!\s*[AaPp][Mm])", military_time_replacer, text
        )

        return text

    # ---------------------------------------------------------------------------
    # Date helpers
    # ---------------------------------------------------------------------------

    def _ordinal_day(self, day: str) -> str:
        """Convert a numeric day ("01"-"31") to its spoken ordinal word ("first", "twenty-third", etc.)."""

        day_num = int(day)

        # Pre-built map for the 31 days of a month – fast and avoids library quirks.
        _ordinal_map = {
            1: "first",
            2: "second",
            3: "third",
            4: "fourth",
            5: "fifth",
            6: "sixth",
            7: "seventh",
            8: "eighth",
            9: "ninth",
            10: "tenth",
            11: "eleventh",
            12: "twelfth",
            13: "thirteenth",
            14: "fourteenth",
            15: "fifteenth",
            16: "sixteenth",
            17: "seventeenth",
            18: "eighteenth",
            19: "nineteenth",
            20: "twentieth",
            21: "twenty-first",
            22: "twenty-second",
            23: "twenty-third",
            24: "twenty-fourth",
            25: "twenty-fifth",
            26: "twenty-sixth",
            27: "twenty-seventh",
            28: "twenty-eighth",
            29: "twenty-ninth",
            30: "thirtieth",
            31: "thirty-first",
        }

        return _ordinal_map.get(day_num, p.ordinal(day_num))

    @lru_cache(maxsize=128)
    def _normalize_dates(self, text):
        """
        Convert date formats to their spoken form.
        Handles common date formats: MM/DD/YYYY, DD/MM/YYYY, YYYY-MM-DD
        """
        # Dictionary of month names
        months = {
            "01": "January",
            "1": "January",
            "02": "February",
            "2": "February",
            "03": "March",
            "3": "March",
            "04": "April",
            "4": "April",
            "05": "May",
            "5": "May",
            "06": "June",
            "6": "June",
            "07": "July",
            "7": "July",
            "08": "August",
            "8": "August",
            "09": "September",
            "9": "September",
            "10": "October",
            "11": "November",
            "12": "December",
        }

        # Convert year to spoken form
        def year_to_words(year):
            # For years between 1000 and 1999
            if len(year) == 4 and year.startswith("1"):
                # Years like 1984 are read as "nineteen eighty-four"
                return f"{self._number_to_words(int(year[:2]))} {self._number_to_words(int(year[2:]))}"
            # For years between 2000 and 2099
            elif len(year) == 4 and year.startswith("2"):
                if year[2:] == "00":
                    # 2000 is read as "two thousand"
                    return "two thousand"
                else:
                    # 2023 is read as "twenty twenty-three"
                    return f"{self._number_to_words(int(year[:2]))} {self._number_to_words(int(year[2:]))}"
            else:
                # Other years read digit by digit
                return self._number_to_words(int(year))

        # Handle format MM/DD/YYYY (US format)
        def us_date_replacer(match):
            month, day, year = match.groups()
            return f"{months[month]} {self._ordinal_day(day)}, {year_to_words(year)}"

        # Handle format DD/MM/YYYY (European format)
        def eu_date_replacer(match):
            day, month, year = match.groups()
            return f"{self._ordinal_day(day)} of {months[month]}, {year_to_words(year)}"

        # Handle ISO format YYYY-MM-DD
        def iso_date_replacer(match):
            year, month, day = match.groups()
            return f"{months[month]} {self._ordinal_day(day)}, {year_to_words(year)}"

        # -------------------------------------------------------------------
        # Apply replacements – order matters
        # -------------------------------------------------------------------

        # US Format: MM/DD/YYYY
        text = re.sub(r"(\d{1,2})/(\d{1,2})/(\d{4})", us_date_replacer, text)

        # European Format: DD/MM/YYYY
        # Since this can conflict with US format, it's commented out
        # Uncomment and use if your application mainly uses European format
        # text = re.sub(r'(\d{1,2})/(\d{1,2})/(\d{4})', eu_date_replacer, text)

        # ISO Format: YYYY-MM-DD
        text = re.sub(r"(\d{4})-(\d{1,2})-(\d{1,2})", iso_date_replacer, text)

        return text

    @lru_cache(maxsize=128)
    def _normalize_urls(self, text):
        """
        Convert URLs to their spoken form.
        Examples:
        - "www.example.com" -> "www dot example dot com"
        - "https://example.com/path" -> "https colon slash slash example dot com slash path"
        """
        # Simple URL replacements
        text = re.sub(
            r"(https?://)",
            lambda m: m.group(1).replace(":", " colon ").replace("/", " slash "),
            text,
        )
        text = re.sub(r"www\.", "www dot ", text)
        text = re.sub(r"\.com", " dot com", text)
        text = re.sub(r"\.org", " dot org", text)
        text = re.sub(r"\.net", " dot net", text)
        text = re.sub(r"\.io", " dot io", text)
        text = re.sub(r"\.gov", " dot gov", text)
        text = re.sub(r"\.edu", " dot edu", text)

        # Replace remaining dots in URLs (more complex detection would be needed for a robust solution)
        text = re.sub(
            r"(\S+)\.(\S+)/", lambda m: f"{m.group(1)} dot {m.group(2)}/", text
        )
        text = re.sub(r"/(\S+)", lambda m: m.group(0).replace("/", " slash "), text)

        return text

    @lru_cache(maxsize=128)
    def _normalize_special_cases(self, text):
        """
        Handle special cases and domain-specific normalizations.
        """
        # Weather-specific phrases
        weather_conditions = {
            "clear sky": "clear skies",
            "partly cloudy": "partly cloudy skies",
            "overcast": "overcast skies",
            r"humidity is (\d+)%": lambda m: f"humidity is {self._number_to_words(int(m.group(1)))} percent",
            r"wind is (\d+(?:\.\d+)?)\s*(km/h|mph)": lambda m: f"wind is {self._normalize_numbers(m.group(1))} {UNIT_MAPPING.get(m.group(2), m.group(2))}",
        }

        for pattern, replacement in weather_conditions.items():
            if callable(replacement):
                text = re.sub(pattern, replacement, text)
            else:
                text = re.sub(pattern, replacement, text)

        return text

    # ---------------------------------------------------------------------------
    # ISO date-time normalizer (e.g., 2025-09-09T00:00:00.000Z)
    # ---------------------------------------------------------------------------

    @lru_cache(maxsize=128)
    def _normalize_iso_datetime(self, text: str) -> str:
        """Convert ISO 8601 date-time strings (basic form returned by APIs) into a
        simpler `YYYY-MM-DD HH:MM` format which downstream *date* and *time*
        normalisers can process correctly.

        Example:
            2025-09-09T00:00:00.000Z  ->  2025-09-09 00:00
        """

        def iso_replacer(match):
            year, month, day, hour, minute = match.groups()
            # If timestamp has no meaningful time (00:00) drop the time component
            if hour == "00" and minute == "00":
                return f"{year}-{month}-{day}"
            return f"{year}-{month}-{day} {hour}:{minute}"

        # Match: YYYY-MM-DDTHH:MM (ignore seconds / millis / Z)
        pattern = r"(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})(?::\d{2}(?:\.\d+)?)?Z?"
        return re.sub(pattern, iso_replacer, text)

    def _clean_response(self, text):
        """
        Main function to clean and normalize text for TTS.
        Applies all normalization functions in sequence.
        """
        # Skip normalization if text is None or empty
        if not text:
            return text

        # Apply normalization steps in sequence
        text = self._normalize_iso_datetime(
            text
        )  # ISO date-time first (simplify structure)
        text = self._normalize_dates(text)  # Convert dates (needs digits intact)
        text = self._normalize_time(text)  # Then convert times
        text = self._normalize_units(text)  # Units include numbers
        text = self._normalize_phone_numbers(
            text
        )  # Phone numbers before generic numbers
        text = self._normalize_numbers(text)  # General number conversion
        text = self._normalize_urls(text)  # URLs after number replacements
        text = self._normalize_special_cases(text)  # Domain / weather / etc.

        return text
