import numpy as np
import scipy
import oracles


def QuadraticOracle():
    # Quadratic function:
    #   f(x) = 1/2 x^T x - [1, 2, 3]^T x
    A = np.eye(3)
    b = np.array([1, 2, 3])
    quadratic = oracles.QuadraticOracle(A, b)

    # Check at point x = [0, 0, 0]
    x = np.zeros(3)
    print(quadratic.grad(x))
    print(np.allclose(quadratic.grad(x), -b))
    # assert_almost_equal(quadratic.func(x), 0.0))


if __name__ == '__main__':
    QuadraticOracle()