"""Text normalization shared by training and inference."""
from __future__ import annotations
import re

_URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
_SPACE_RE = re.compile(r"\s+")

def clean_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).replace("\x00", " ")
    text = _URL_RE.sub(" URLTOKEN ", text)
    text = _SPACE_RE.sub(" ", text)
    return text.strip()
