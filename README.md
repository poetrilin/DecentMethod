
# Gradient Descent and Newton's Method's inplementation

- refer : [Convex Optimization – Boyd and Vandenberghe](https://web.stanford.edu/~boyd/cvxbook/)

take Logistic Regression and Linear Least Squares as examples

### File structure

```shell
├── Optimizer.py   # Optimizer class,base class of all optimizers
├── GD.py         # Gradient Descent
├── Newton.py     # Newton's method  
├── README.md
```

### Grad and Hessian's calculation

- Linear Least Squares
  - values: $f(x):=\sum_{i=1}^{N} \| a_{i}^{T} x-b_{i} \|_{2}^{2}= \| A x-b \|_{2}^{2} $
  - gradient: $\nabla f(x)=2 A^{T}(A x-b)$
  - Hessin: $H=2 A^{T} A$

- Logistic Regression
  - values: $f(x):=\sum_{i=1}^{N} \log \left(1+e^{ a_{i}^{T} x} \right)-(1-b)a_i^Tx= \mathbf{1}^{T} \log \left(1+e^{A x} \right)-(1-b)^{T} A x$
  - gradient: $\nabla f(x)= A^{T} \left( \frac{e^{A x}}{1+e^{A x}} \right)-A^{T} (1-b)$
  - Hessin: $H(x)=A^{T}\textbf{diag}( \frac{e^{A x}}{(1+e^{A x})^2})A$

some details are shown below:

$$
\begin{aligned}
f(x) &=\sum_{i=1}^{N} \log \left(1+e^{A_{[i,:]} x} \right)-(1-b)A_{[i,:]}x \\
\frac{\partial f(x)}{\partial x_j} &=\sum_{i=1}^{N} \frac{e^{A_{[i,:]} x}}{1+e^{A_{[i,:]} x}}A_{[i,j]}-A_{[i,j]}(1-b) \\
&=\left (A^T\frac{e^{A x}}{1+e^{A x}}-A^T(1-b)\right ) _j \\
\end{aligned}
$$
故 $\nabla f(x)= A^{T} \left( \frac{e^{A x}}{1+e^{A x}} \right)-A^{T} (1-b)$

$$
\begin{aligned}
H_{ij} &=\frac{\partial^2 f(x)}{\partial x_i \partial x_j} \\
&=\sum_{i=1}^{N} \frac{\partial}{\partial x_j} \left( \frac{e^{A_{[i,:]} x}}{1+e^{A_{[i,:]} x}}A_{[i,i]}-A_{[i,i]}(1-b) \right) \\
&=\sum_{i=1}^{N} A_{[i,j]}\frac{e^{A_{[i,:]} x}}{(1+e^{A_{[i,:]} x})^2}A_{[i,j]} \\
&=\left (A^T\textbf{diag}( \frac{e^{A x}}{(1+e^{A x})^2})A\right )_{ij}
\end{aligned}
$$