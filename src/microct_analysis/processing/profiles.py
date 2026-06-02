"""Scanner profile registry for segmentation calibration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ScannerProfile:
    key: str
    manufacturer_substrings: tuple[str, ...] = ()
    profile_mask_threshold: float | None = None
    profile_marker_threshold: float | None = None
    has_documented_thresholds: bool = False

    def matches(self, manufacturer: str, model: str) -> bool:
        text = f"{manufacturer or ''} {model or ''}".lower()
        return any(substring in text for substring in self.manufacturer_substrings)


SCANCO = ScannerProfile(
    key="scanco",
    manufacturer_substrings=("scanco",),
    profile_mask_threshold=2500.0,
    profile_marker_threshold=3500.0,
    has_documented_thresholds=True,
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
