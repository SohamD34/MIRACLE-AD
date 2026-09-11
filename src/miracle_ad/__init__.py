"""MIRACLE-AD: raw-speech multiple-instance learning for AD detection."""

__version__ = "0.1.0"

from .models import BackboneType, NetworkType, available_backbones, available_networks, get_model

__all__ = [
    "BackboneType",
    "NetworkType",
    "available_backbones",
    "available_networks",
    "get_model",
]
