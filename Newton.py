import numpy as np
from Optimizer import Optimizer


class NewtonOptimizer(Optimizer):
    def __init__(self, starting_point: np.ndarray, *, tol: float = 1e-14, max_iter: int = 30,
                 alpha: float = 0.1, beta: float = 0.5, verbose: bool = True):
        """
        Initialize the Newton optimizer.
        @param starting_point: np.ndarray
            The starting point for optimization.
        @param tol: float
            The tolerance for the gradient norm to determine convergence.
        """
        super().__init__(starting_point, tol=tol, verbose=True)
        self.x = starting_point
        self.tol = tol
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta

    def gradient(self, x: np.array = None) -> np.ndarray:
        raise NotImplementedError("Subclasses should implement this method.")

    def hessian(self, x: np.array = None) -> np.ndarray:
        raise NotImplementedError("Subclasses should implement this method.")

    def value(self, x: np.ndarray) -> float:
        raise NotImplementedError("Subclasses should implement this method.")

    def optimize(self) -> np.ndarray:
        """
        Perform the Newton optimization.
        """
        for k in range(self.max_iter):
            if self.verbose:
                self.f_history.append(self.value())
                self.x_history.append(self.x.copy())
            # Step 1: Compute the Newton step and decrement
            grad = self.gradient()
            hess = self.hessian()
            hess_inv = np.linalg.inv(hess)
            newton_step = - hess_inv @ grad
            # lambda^2 = g^T * H^-1 * g
            hessian_quadratic = - grad.T @ newton_step
            # Step 2: Stopping criterion
            if hessian_quadratic / 2 <= self.tol:
                break
            # Step 3: Line search
            # to avoid duplicate computation lambda^2
            t = 1
            while self.value(self.x + t * newton_step) > self.value() - self.alpha * t * hessian_quadratic:
                t *= self.beta
            # Step 4: Update
            self.x += t * newton_step

        self.optimal_value = self.value()
        self.iter_count = k

    def __repr__(self):
        return f"NewtonOptimizer(x={self.x}, tol={self.tol})"

# Example subclass for linear regression using Newton's method


class LinearRegressionNewton(NewtonOptimizer):
    """
    loss function: f(x) = ||Ax - b||^2
    """

    def __init__(self, A: np.ndarray, b: np.ndarray, starting_point: np.ndarray = None, **kwargs):
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
        return 2 * np.dot(self.A.T, (np.dot(self.A, x) - self.b))

    def value(self, x: np.array = None) -> float:
        if x is None:
            x = self.x
        return np.sum((np.dot(self.A, x) - self.b) ** 2)

    def hessian(self) -> np.ndarray:
        H = 2 * (self.A.T @ self.A)
        # 确保H正定
        if np.linalg.eigvals(H).min() < 0:
            raise ValueError("Hessian matrix is not positive definite.")
        return H


class LogisticRegressionNewton(NewtonOptimizer):
    """
    Loss = np.sum( logistic(A @ x), axis=0)- ((1-y).T)@(A @ x)
    """

    def __init__(self, A, y, starting_point=None, epsilon: float = 1e-10, **kwargs):
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

        return np.matmul(self.A.T, np.exp(self.A@self.x)/(1 + np.exp(self.A@self.x)))-self.A.T@(1-self.y)

    def value(self, x: np.array = None) -> float:
        if x is None:
            x = self.x

        return np.sum(np.log(1 + np.exp(self.A@self.x)+self.epsilon)) - (1-self.y) @ self.A@self.x

    def hessian(self, x: np.array = None) -> np.ndarray:
        if x is None:
            x = self.x
        ax = np.clip(self.A @ self.x, -100, 100)
        tmp = np.exp(ax)
        return self.A.T @ np.diag(tmp/np.dot(1+tmp, 1+tmp)) @ self.A

    def __repr__(self) -> str:
        return f"LogisticRegressionNewton(A={self.A}, y={self.y}, starting_point={self.x})"
