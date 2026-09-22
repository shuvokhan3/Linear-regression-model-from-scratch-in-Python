"""
Automatic differentiation utilities for MiniTorch.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Sequence


def central_difference(
    f: Callable[..., float],
    *vals: float,
    arg: int = 0,
    epsilon: float = 1e-6,
) -> float:
    """
    Compute the numerical derivative of f with respect to argument `arg`.

    Uses the central difference formula:

        (f(x + h) - f(x - h)) / (2h)

    Args:
        f: Function to differentiate.
        *vals: Input values to f.
        arg: Which argument to differentiate with respect to (0-indexed).
        epsilon: Step size for numerical differentiation.

    Returns:
        Approximate derivative.

    Example:
        >>> def mul(x, y):
        ...     return x * y
        >>> central_difference(mul, 3, 4, arg=0)
        4.0
        >>> central_difference(mul, 3, 4, arg=1)
        3.0
    """

    vals_list = list(vals)

    # Create values with the selected argument increased by epsilon.
    vals_plus = vals_list.copy()
    vals_plus[arg] += epsilon

    # Create values with the selected argument decreased by epsilon.
    vals_minus = vals_list.copy()
    vals_minus[arg] -= epsilon

    # Compute function values.
    f_plus = f(*vals_plus)
    f_minus = f(*vals_minus)

    # Central difference.
    return (f_plus - f_minus) / (2 * epsilon)


@dataclass
class Variable:
    """
    A node in the computation graph.

    Attributes:
        history:
            Record of the operation that created this variable.

        derivative:
            Accumulated gradient during the backward pass.

        name:
            Optional name used for debugging.

        requires_grad:
            Whether this variable should receive gradients.
    """

    history: Optional["History"] = None
    derivative: Optional[float] = None
    name: Optional[str] = None
    requires_grad: bool = False

    def is_leaf(self) -> bool:
        """
        Return True if this variable was not created by an operation.
        """
        return self.history is None

    def is_constant(self) -> bool:
        """
        Return True if this variable does not require gradients.
        """
        return not self.requires_grad

    def requires_grad_(self, requires_grad: bool = True) -> "Variable":
        """
        Set whether this variable should track gradients.

        Args:
            requires_grad:
                Whether gradients should be tracked.

        Returns:
            This variable.
        """
        self.requires_grad = requires_grad
        return self


@dataclass
class History:
    """
    Record the operation that created a variable.

    Attributes:
        last_fn:
            The function class that created this variable.

        ctx:
            Context object storing values needed during backward.

        inputs:
            Input variables to the operation.
    """

    last_fn: Optional[type] = None
    ctx: Optional["Context"] = None
    inputs: Sequence[Variable] = ()
    