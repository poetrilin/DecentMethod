import numpy as np
import time
import matplotlib.pyplot as plt


class Optimizer:
    """
    base class for optimizers
    """

    def __init__(self, starting_point: np.ndarray, *, tol: float = 1e-8,
                 alpha: float = 0.1, beta: float = 0.5, verbose: bool = True):
        """
        @param:
        starting_point
        tol : tolerance, bound the result of the gradient norm.
        """
        assert alpha > 0 and alpha < 0.5, "alpha should be in (0, 0.5)"
        assert beta > 0 and beta < 1, "beta should be in (0, 1)"
        self.x = starting_point
        self.tol = tol
        self.x_history = []
        self.f_history = []
        self.iter_count = 0
        self.optimal_value = None
        self.verbose = verbose
        self.alpha = alpha
        self.beta = beta

    def optimize(self):
        raise NotImplementedError

    def reset(self):
        self.x_history = []
        self.f_history = []
        self.iter_count = 0

    def backtracking_line_search(self, gradient: np.array, step: np.array) -> float:
        """
        let f(x + t * step) <= f(x) + t * alpha * ∇f(x)^T * step
        default: alpha = 0.1, beta = 0.5, step = -\nabla f(x)
        """
        t = 1
        new_x = self.x + t * step
        while self.value(new_x) > self.value(self.x) - t * self.alpha * np.dot(step.T, gradient):
            t *= self.beta
            new_x = self.x + t * step
        return t

    def exact_line_search(self, gradient: np.array, step: np.array) -> float:
        raise NotImplementedError("TODO")

    def __call__(self):
        start_time = time.time()
        self.optimize()
        return time.time() - start_time

    def plot_convergence(self):
        # Plotting the convergence of the optimizer
        assert self.verbose, "assure  the verbose mode is True"
        assert len(
            self.x_history) > 0, "No history to plot. Run the optimizer first."
        plt.figure(figsize=(14, 7))

        plt.subplot(1, 2, 1)
        plt.plot(np.arange(1, len(self.x_history) + 1),
                 [max(np.linalg.norm(x - self.x), 1e-20) for x in self.x_history], label='Solution error')
        plt.xlabel('Iteration')
        plt.ylabel('$||x^{(k)} - x^*||_2$')
        plt.yscale("log")
        plt.title('Convergence of Optimal x')
        plt.legend()

        plt.subplot(1, 2, 2)
        f_history = np.array(self.f_history)
        p_star = self.optimal_value
        f_history = np.abs(f_history - p_star)
        plt.plot(np.arange(1, len(f_history)),
                 f_history[:-1], label='Objective function')
        plt.xlabel('Iteration')
        plt.ylabel('$|f(x^{(k)}) - p^*|$')
        plt.yscale("log")
        plt.title('Objective Function Convergence')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def report_runtime(self, eps):
        """Report the runtime required to reach the specified RMSE."""
        d = self.x.shape[0]
        for x_k, f_k in zip(self.x_history, self.f_history[self.iter_count+1:]):
            if np.sqrt(d) * np.linalg.norm(x_k - self.x_history[0]) <= eps:
                print(
                    f"Runtime to reach RMSE {eps:.2e}: {self.runtime:.4f} seconds")
                break
