import pytest 
import math

from hypothesis import given
from hypothesis import strategies as st
from minitorch.operators import mul , id ,add, neg, lt, eq,max,is_close,sigmoid,relu,log,exp,inv,log_back,inv_back,relu_back,map,zipWith,reduce,negList,addLists,sum,prod

def test_mul():
    assert mul(5, 5) == 25
    assert mul(2, 3) == 6 
    assert mul(-2, 3) == -6
    assert mul(-2, -3) == 6
    assert mul(0, 10) == 0

def test_id():
    assert id(4) == 4
    assert id(-5) == -5
    assert id(0) == 0
    assert id(3.14) == 3.14

def test_add():
    assert add(2, 3) == 5
    assert add(-2, 3) == 1
    assert add(-2, -3) == -5
    assert add(0, 5) == 5
    assert add(1.5, 2.5) == 4.0

def test_neg():
    assert neg(5) == -5
    assert neg(-5) == 5
    assert neg(0) == 0
    assert neg(3.14) == -3.14

def test_lt():
    assert lt(2, 5) == 1.0
    assert lt(5, 2) == 0.0
    assert lt(5, 5) == 0.0
    assert lt(-5, -2) == 1.0

def test_eq():
    assert eq(5, 5) == 1.0
    assert eq(5, 3) == 0.0
    assert eq(-2, -2) == 1.0
    assert eq(0, 1) == 0.0

def test_max():
    assert max(5, 3) == 5
    assert max(3, 5) == 5
    assert max(5, 5) == 5
    assert max(-2, -5) == -2
    assert max(-5, -2) == -2

def test_is_close():
    assert is_close(1.0, 1.0) == 1.0
    assert is_close(1.0, 1.005) == 1.0
    assert is_close(1.0, 1.02) == 0.0
    assert is_close(-1.0, -1.005) == 1.0






# property test 

"""Property test for log_back"""
@given(
    st.floats(
        min_value=0.001,
        max_value=1000,
        allow_nan=False,
        allow_infinity=False,
    ),
    st.floats(
        min_value=-1000,
        max_value=1000,
        allow_nan=False,
        allow_infinity=False,
    ),
)
def test_log_back(x, grad):
    assert log_back(x, grad) == pytest.approx(grad / x)

"""Property test for inv_back"""
@given(
    st.floats(
        min_value=-1000,
        max_value=1000,
        allow_nan=False,
        allow_infinity=False,
    ).filter(lambda x: abs(x) > 1e-6),
    st.floats(
        min_value=-1000,
        max_value=1000,
        allow_nan=False,
        allow_infinity=False,
    ),
)
def test_inv_back(x, grad):
    assert inv_back(x, grad) == pytest.approx(-grad / (x * x))

"""Property test for relu_back"""
@given(
    st.floats(
        min_value=0.001,
        max_value=1000,
        allow_nan=False,
        allow_infinity=False,
    ),
    st.floats(
        min_value=-1000,
        max_value=1000,
        allow_nan=False,
        allow_infinity=False,
    ),
)
def test_relu_back_positive(x, grad):
    assert relu_back(x, grad) == grad

"""relu_back_negative"""
@given(
    st.floats(
        min_value=-1000,
        max_value=-0.001,
        allow_nan=False,
        allow_infinity=False,
    ),
    st.floats(
        min_value=-1000,
        max_value=1000,
        allow_nan=False,
        allow_infinity=False,
    ),
)
def test_relu_back_negative(x, grad):
    assert relu_back(x, grad) == 0.0


def test_relu_back_zero():
    assert relu_back(0.0, 10.0) == 0.0





"""Property test for sigmoid"""
@given(
    st.floats(
        min_value=-20,
        max_value=20,
        allow_nan=False,
        allow_infinity=False,
    )
)
def test_sigmoid_range(x):
    result = sigmoid(x)

    assert 0.0 < result < 1.0


def test_sigmoid_zero():
    assert sigmoid(0.0) == pytest.approx(0.5)


@given(
    st.floats(
        min_value=-20,
        max_value=20,
        allow_nan=False,
        allow_infinity=False,
    )
)
def test_sigmoid_symmetry(x):
    assert sigmoid(-x) == pytest.approx(1.0 - sigmoid(x))


@given(
    st.floats(min_value=-20, max_value=20, allow_nan=False),
    st.floats(min_value=-20, max_value=20, allow_nan=False),
)
def test_sigmoid_monotonic(x, y):
    if x < y:
        assert sigmoid(x) <= sigmoid(y)





"""Property test for relu"""
@given(
    st.floats(
        min_value=-1000,
        max_value=1000,
        allow_nan=False,
        allow_infinity=False,
    )
)
def test_relu_non_negative(x):
    assert relu(x) >= 0.0


@given(
    st.floats(
        min_value=0.0,
        max_value=1000,
        allow_nan=False,
        allow_infinity=False,
    )
)
def test_relu_positive(x):
    assert relu(x) == x


@given(
    st.floats(
        min_value=-1000,
        max_value=0.0,
        allow_nan=False,
        allow_infinity=False,
    )
)
def test_relu_non_positive(x):
    assert relu(x) == 0.0





"""Property test for log"""
@given(
    st.floats(
        min_value=-20,
        max_value=20,
        allow_nan=False,
        allow_infinity=False,
    )
)
def test_log_exp_inverse(x):
    assert log(exp(x)) == pytest.approx(x)


def test_log_one():
    assert log(1.0) == pytest.approx(0.0)

@given(
    st.floats(min_value=0.001, max_value=1000),
    st.floats(min_value=0.001, max_value=1000),
)
def test_log_product(a, b):
    assert log(a * b) == pytest.approx(log(a) + log(b))





"""Property test for exp"""
@given(
    st.floats(min_value=0.001, max_value=1000),
    st.floats(min_value=0.001, max_value=1000),
)
def test_log_product(a, b):
    assert log(a * b) == pytest.approx(log(a) + log(b))

def test_exp_zero():
    assert exp(0.0) == pytest.approx(1.0)

@given(
    st.floats(min_value=-10, max_value=10, allow_nan=False),
    st.floats(min_value=-10, max_value=10, allow_nan=False),
)
def test_exp_addition(a, b):
    assert exp(a + b) == pytest.approx(exp(a) * exp(b))


"""Property test for inv"""
@given(
    st.floats(
        min_value=-1000,
        max_value=1000,
        allow_nan=False,
        allow_infinity=False,
    ).filter(lambda x: abs(x) > 1e-6)
)
def test_inv_multiplicative_inverse(x):
    assert x * inv(x) == pytest.approx(1.0)

@given(
    st.floats(
        min_value=-1000,
        max_value=1000,
        allow_nan=False,
        allow_infinity=False,
    ).filter(lambda x: abs(x) > 1e-6)
)
def test_inv_twice(x):
    assert inv(inv(x)) == pytest.approx(x)





#test higher order function
def test_map():
    result = map(lambda x: x * 2)([1, 2, 3, 4])
    assert result == [2, 4, 6, 8]


def test_zipWith():
    result = zipWith(lambda x, y: x + y)([1, 2, 3], [4, 5, 6])
    assert result == [5, 7, 9]


def test_reduce():
    result = reduce(lambda x, y: x + y, 0)([1, 2, 3, 4])
    assert result == 10

def test_negList():
    result = negList([1, 2, 3, 4])
    assert result == [-1, -2, -3, -4]


def test_addLists():
    result = addLists([1, 2, 3], [4, 5, 6])
    assert result == [5, 7, 9]


def test_sum():
    result = sum([1, 2, 3, 4])
    assert result == 10

def test_prod():
    result = prod([1, 2, 3, 4])
    assert result == 24