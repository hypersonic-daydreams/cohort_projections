"""
FIPS code normalization utilities.

Provides consistent zero-padded string conversion for FIPS codes that may
arrive as integers, floats (``38001.0``), or already-formatted strings.
"""

from __future__ import annotations

import math


def normalize_fips(value: object, width: int = 5) -> str:
    """Normalize a FIPS-like value to a zero-padded digit string.

    Strips whitespace, removes a trailing ``.0`` suffix (common when FIPS
    codes round-trip through Excel/CSV as floats), extracts only digit
    characters, and zero-pads or truncates to *width*.

    Args:
        value: Any scalar that can be cast to ``str`` -- typically an int,
            float, or string FIPS code.  Must not be ``None`` or ``NaN``.
        width: Desired output length (e.g. 2 for state, 3 for county-part,
            5 for full county, 7 for place).

    Returns:
        A string of exactly *width* digit characters.

    Raises:
        ValueError: If *value* is ``None``, ``NaN``, or contains no digits.
    """
    if value is None:
        raise ValueError("normalize_fips received None")
    # Handle NaN for both float and numpy types
    try:
        if math.isnan(float(value)):  # type: ignore[arg-type]
            raise ValueError("normalize_fips received NaN")
    except (TypeError, ValueError):
        # Not a numeric type that could be NaN -- fine, continue
        if isinstance(value, float) and math.isnan(value):
            raise ValueError("normalize_fips received NaN")  # noqa: B904

    text = str(value).strip().removesuffix(".0")
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        raise ValueError(f"normalize_fips could not extract digits from {value!r}")
    return digits.zfill(width)[-width:]


def normalize_fips_optional(value: object | None, width: int = 5) -> str | None:
    """Normalize a FIPS-like value, returning ``None`` for null inputs.

    Accepts ``None`` and ``NaN`` gracefully (returns ``None``).  For all
    other inputs delegates to :func:`normalize_fips`.

    Args:
        value: A scalar FIPS code, or ``None``/``NaN``.
        width: Desired output length.

    Returns:
        A zero-padded digit string of *width* characters, or ``None`` if
        the input was null-ish or contained no digits.
    """
    if value is None:
        return None
    try:
        if math.isnan(float(value)):  # type: ignore[arg-type]
            return None
    except (TypeError, ValueError):
        if isinstance(value, float) and math.isnan(value):
            return None

    text = str(value).strip().removesuffix(".0")
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return None
    return digits.zfill(width)[-width:]
