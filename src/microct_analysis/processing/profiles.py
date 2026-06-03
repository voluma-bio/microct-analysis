"""Scanner profile registry for segmentation calibration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ScannerProfile:
    key: str
    manufacturer_substrings: tuple[str, ...] = ()

    def matches(self, manufacturer: str, model: str) -> bool:
        text = f"{manufacturer or ''} {model or ''}".lower()
        return any(substring in text for substring in self.manufacturer_substrings)


SCANCO = ScannerProfile(
    key="scanco",
    manufacturer_substrings=("scanco",),
)
UNKNOWN = ScannerProfile(key="unknown")

_PROFILES: tuple[ScannerProfile, ...] = (SCANCO,)


def detect(manufacturer: str, model: str = "") -> ScannerProfile:
    """Return the matching scanner profile, or the unknown fallback."""

    for profile in _PROFILES:
        if profile.matches(manufacturer, model):
            return profile
    return UNKNOWN


def get(key: str) -> ScannerProfile:
    """Return a scanner profile by explicit key."""

    for profile in (*_PROFILES, UNKNOWN):
        if profile.key == key:
            return profile
    raise ValueError(f"unknown scanner profile key: {key!r}")
