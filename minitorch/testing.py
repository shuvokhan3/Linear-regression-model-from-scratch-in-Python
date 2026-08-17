"""Testing utilities for MiniTorch."""

def assert_close(a: float, b: float) -> None:
    """Assert two floats are close within tolerance."""
    assert abs(a - b) < 1e-2, f"Values not close: {a} vs {b}"
