import numpy as np


def cholesky_decomp(matrix):
    """
    Performs Choleksy Decomposition on a matrix. Returns lower triangle L.
    """
    L = np.zeros_like(matrix, dtype=float)
    for i in range(matrix.shape[0]):
        for j in range(i + 1):
            if i == j:
                sum = 0
                for k in range(j):
                    sum += L[j][k] ** 2
                L[j][j] = np.sqrt(matrix[j][j] - sum)
            else:
                sum = 0
                for k in range(j):
                    sum += L[i][k] * L[j][k]
                L[i][j] = (matrix[i][j] - sum) / L[j][j]
    return L


if __name__ == "__main__":
    # Test 1: 2x2 example
    A1 = np.array([[4, 2], [2, 3]], dtype=float)
    L1 = cholesky_decomp(A1)
    assert np.allclose(L1 @ L1.T, A1)

    # Test 2: Identity matrix
    A2 = np.eye(3)
    L2 = cholesky_decomp(A2)
    assert np.allclose(L2 @ L2.T, A2)

    # Test 3: Diagonal positive definite
    A3 = np.diag([5, 6, 7])
    L3 = cholesky_decomp(A3)
    assert np.allclose(L3 @ L3.T, A3)

    # Test 4: 3x3 symmetric positive definite
    A4 = np.array([[25, 15, -5], [15, 18, 0], [-5, 0, 11]], dtype=float)
    L4 = cholesky_decomp(A4)
    assert np.allclose(L4 @ L4.T, A4)

    # Test 5: Random SPD matrix (A = B @ B.T ensures SPD)
    np.random.seed(0)
    B = np.random.randn(4, 4)
    A5 = B @ B.T
    L5 = cholesky_decomp(A5)
    assert np.allclose(L5 @ L5.T, A5)

    print("All tests passed!")
    A = np.array([[1, 0.2, 0.9], [0.2, 1, 0.5], [0.9, 0.5, 1]], dtype=float)
    L = cholesky_decomp(A)
    print(L)
