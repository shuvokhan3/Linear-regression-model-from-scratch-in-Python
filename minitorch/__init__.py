from .autodiff import central_difference
from .scalar import Scalar





a = Scalar(2.0, name="a")
b = Scalar(3.0, name="b")

print(f"a.data = {a.data}")
print(f"a.is_leaf() = {a.is_leaf()}")
print(f"a.derivative = {a.derivative}")

a.accumulate_derivative(1.0)
a.accumulate_derivative(2.0)
print(f"After accumulating: a.derivative = {a.derivative}")