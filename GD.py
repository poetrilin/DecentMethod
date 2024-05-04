import numpy as np
from Optimizer import Optimizer


class GradientDescentOptimizer(Optimizer):
    def __init__(self, starting_point: np.ndarray, *, learning_rate: float = 1e-2,
                 max_iter: int = 3000, tol: float = 1e-10):
        """
        @param:
        starting_point
        max_iter 
        tol : tolerance, bound the result of the gradient norm.
        """
        super().__init__(starting_point, tol=tol, verbose=True)
        self.x = starting_point
        self.max_iter = max_iter
        self.tol = tol

    def gradient(self, x: np.array = None):
        raise NotImplementedError("Subclasses should implement this method.")

    def value(self, x: np.array = None):
        raise NotImplementedError("Subclasses should implement this method.")

    def optimize(self):
        """
        Algorithm 9.1 Gradient Descent.
        Linear search 采用 Alg9.2 backtracking line search.
        """
        for k in range(self.max_iter):
            if self.verbose:
                self.f_history.append(self.value())
                self.x_history.append(self.x.copy())
            gradient = self.gradient()

            step = - gradient  # Step 1: ∆x := −∇f(x)
            t = self.backtracking_line_search(
                gradient, step)  # Step 2: Line search
            self.x += t * step  # Step 3: Update

            if np.linalg.norm(gradient) < self.tol:
                if self.verbose:
                    print(f"Optimizer converged after {k+1} iterations.")
                break
        self.iter_count = k
        self.optimal_value = self.value()

    def __repr__(self) -> str:
        return f"GradientDescentOptimizer(starting_point={self.x})"


class LinearRegressionGD(GradientDescentOptimizer):
    """
    Loss = ||Ax - b||^2
    """

    def __init__(self, A: np.array, b: np.array, starting_point=None, **kwargs):
        super().__init__(starting_point, **kwargs)
        assert A.shape[0] == b.shape[0], "A and b should have the same number of rows."
        self.A = A
        self.b = b
        if starting_point is None:
            self.x = np.zeros(A.shape[1])
        else:
            self.x = starting_point

    def gradient(self, x: np.array = None
                 ) -> np.array:
        if x is None:
            x = self.x
        return 2 * self.A.T @ (self.A @ x - self.b)

    def value(self, x: np.array = None) -> float:
        if x is None:
            x = self.x
        return np.sum(np.square(self.A @ x - self.b))

    def __repr__(self) -> str:
        return f"LinearRegressionGD(A={self.A}, b={self.b}, starting_point={self.x})"


class LogisticRegressionGD(GradientDescentOptimizer):
    """
    Loss = np.sum( logistic(A @ x), axis=0)- ((1-y).T)@(A @ x)
    """

    def __init__(self, A, y, starting_point=None, epsilon: float = 1e-12, **kwargs):
        super().__init__(starting_point, **kwargs)
        assert A.shape[0] == y.shape[0], "A and y should have the same number of rows."
        self.A = A
        assert 0 <= y.min() and y.max(
        ) <= 1, "for logistic, y should be in [0, 1]."
        self.y = y
        if starting_point is None:
            self.x = np.zeros(A.shape[1])
        else:
            self.x = starting_point
        self.epsilon = min(epsilon, self.tol)

    def gradient(self, x: np.array = None) -> np.array:
        if x is None:
            x = self.x
        assert x.shape[0] == self.A.shape[1], "x should have the same number of columns as A."
        return np.matmul(self.A.T, np.exp(self.A@self.x)/(1 + np.exp(self.A@self.x)))-self.A.T@(1-self.y)

    def value(self, x: np.array = None) -> float:
        if x is None:
            x = self.x
        assert x.shape[0] == self.A.shape[1], "x should have the same number of columns as A."

        return np.sum(np.log(1 + np.exp(self.A@self.x))) - (1-self.y) @ self.A@self.x

    def __repr__(self) -> str:
        return f"LogisticRegressionGD(X={self.X}, y={self.y}, starting_point={self.x})"
