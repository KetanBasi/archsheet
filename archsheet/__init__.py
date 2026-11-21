"""archsheet — blueprint-style architecture sheets of PyTorch models.

Specs may import the two tracing helpers directly::

    from archsheet import label, phase
"""
try:
    import torch  # noqa: F401
except ImportError:
    raise ImportError(
        "archsheet needs PyTorch in the Python environment that can import your model "
        "(the one you run the project with). Install it there, e.g. `pip install torch`; "
        "`pip install cairosvg` is only needed for --png.") from None

from .trace import label, phase

__all__ = ["label", "phase"]
__version__ = "0.1.0"
