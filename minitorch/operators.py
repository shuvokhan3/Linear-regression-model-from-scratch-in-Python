##operators.py is essentially your mathematical building-block layer.

"""
Mathematical operators for Minitorch .
These form the foundation of all neural network operation
"""

import math
from typing import Callable, Iterable

#Basic Arithmetice Operators

def mul(x:float, y:float) -> float :
    """Multiply two numbers."""
    return x * y

def id(x: float) -> float:
    return x

def add(x: float, y:float) -> float:
    return x + y

def neg(x: float) -> float:
    return -x

def lt(x:float, y:float) -> float:
    return 1.0 if x < y else 0.0

def eq(x: float, y: float) -> float:
    return 1.0 if x == y else 0.0

def max(x: float, y: float) -> float:
    return x if x > y else y

def is_close(x: float, y: float) -> float:
    return 1.0 if abs(x - y) < 1e-2 else 0.0

def sigmoid(x: float) -> float:
    if x >= 0 :
        result = 1.0 / (1.0 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        result = exp_x / (1.0 + exp_x)

    eps = 1e-12
    if result == 1.0:
        return 1.0 - eps
    if result == 0.0:
        return eps
    return result

def relu(x : float) -> float:
    return x if x > 0.0 else 0.0

def log(x: float) -> float:
    return math.log(x)

def exp(x: float) -> float:
    return math.exp(x)

def inv(x: float) -> float:
    return 1.0 / x

def log_back(x: float, grad: float) -> float:
    return grad / x

def inv_back(x: float, grad:float) -> float:
    return -grad / (x * x)

def relu_back(x : float, grad: float) -> float:
    return grad if x > 0 else 0.0


    




