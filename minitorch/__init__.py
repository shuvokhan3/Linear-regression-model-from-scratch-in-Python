from .autodiff import central_difference
from .scalar import Scalar
from minitorch.scalar import Scalar



a = Scalar(2.0, name="a")
b = Scalar(3.0, name="b")

print(f"a.data = {a.data}")
print(f"a.is_leaf() = {a.is_leaf()}")
print(f"a.derivative = {a.derivative}")

a.accumulate_derivative(1.0)
a.accumulate_derivative(2.0)
print(f"After accumulating: a.derivative = {a.derivative}")





# test for ,forward computation + computation history
a = Scalar(2.0)
a.requires_grad_(True)

b = Scalar(3.0)
b.requires_grad_(True)

c = a * b

print(f"c.data = {c.data}")
print(f"c.is_leaf() = {c.is_leaf()}")
print(f"c.history.last_fn = {c.history.last_fn}")