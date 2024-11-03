"""Collection of the core mathematical operators used throughout the code base."""

import math
from typing import Callable, Sequence


def mul(x: float, y: float) -> float:
    """Multiplication function"""
    return x * y


def id(x: float) -> float:
    """Identity function"""
    return x


def add(x: float, y: float) -> float:
    """Sum function"""
    return x + y


def neg(x: float) -> float:
    """Negate function"""
    return -x


def lt(x: float, y: float) -> float:
    """Is lower than function"""
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Is equal function"""
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Max function"""
    return x if x >= y else y


def is_close(x: float, y: float) -> float:
    """Is close function"""
    return 1.0 if abs(x - y) < 1e-2 else 0.0


def sigmoid(x: float) -> float:
    """Sigmoid function"""
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Relu function"""
    return x if x >= 0 else 0.0


def log(x: float) -> float:
    """Logarithm function"""
    return math.log(x + 1e-6)


def exp(x: float) -> float:
    """Exponent function"""
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    """Logarithm derivative function"""
    return d / x


def inv(x: float) -> float:
    """Inverse function"""
    return 1 / x


def inv_back(x: float, d: float) -> float:
    """Inverse derivative function"""
    return -d / x**2


def relu_back(x: float, d: float) -> float:
    """Relu derivative function"""
    return d if x >= 0 else 0.0


def map(fn: Callable[[float], float]) -> Callable[[Sequence[float]], Sequence[float]]:
    """Map function"""
    return lambda ls: [fn(x) for x in ls]


def negList(ls: Sequence[float]) -> Sequence[float]:
    """Negate list function"""
    return map(neg)(ls)


def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Sequence[float], Sequence[float]], Sequence[float]]:
    """Zip and map"""
    return lambda ls1, ls2: [fn(ls1[i], ls2[i]) for i in range(len(ls1))]


def addLists(ls1: Sequence[float], ls2: Sequence[float]) -> Sequence[float]:
    """Sum of two lists"""
    return zipWith(add)(ls1, ls2)


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Sequence[float]], float]:
    """Reduce function"""

    def f(ls: Sequence[float]) -> float:
        res = start
        for val in ls:
            res = fn(res, val)
        return res

    return f


def sum(ls: Sequence[float]) -> float:
    """Sum of list"""
    return reduce(add, 0.0)(ls)


def prod(ls: Sequence[float]) -> float:
    """Product of list"""
    return reduce(mul, 1.0)(ls)
