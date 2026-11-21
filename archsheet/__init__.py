"""archsheet — blueprint-style architecture sheets of PyTorch models.

Specs may import the two tracing helpers directly::

    from archsheet import label, phase
"""
from .trace import label, phase

__all__ = ["label", "phase"]
__version__ = "0.1.0"
