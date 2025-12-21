# Linear Algebra

## Matrices

A matrix $A$ of size $m \times n$:
$$A = \begin{pmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{pmatrix}$$

### Matrix Operations

**Matrix Multiplication**: $(AB)_{ij} = \sum_{k=1}^{n} a_{ik}b_{kj}$

**Determinant** (2×2):
$$\det \begin{pmatrix} a & b \\ c & d \end{pmatrix} = ad - bc$$

**Inverse** (2×2):
$$A^{-1} = \frac{1}{\det(A)} \begin{pmatrix} d & -b \\ -c & a \end{pmatrix}$$

## Eigenvalues and Eigenvectors

For matrix $A$ and vector $\mathbf{v}$:
$$A\mathbf{v} = \lambda\mathbf{v}$$

Where $\lambda$ is the eigenvalue and $\mathbf{v}$ is the eigenvector.

### Characteristic Polynomial
$$\det(A - \lambda I) = 0$$

## Vector Spaces

### Dot Product
$$\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i = |\mathbf{a}||\mathbf{b}|\cos\theta$$

### Cross Product (3D)
$$\mathbf{a} \times \mathbf{b} = \begin{vmatrix}
\mathbf{i} & \mathbf{j} & \mathbf{k} \\
a_1 & a_2 & a_3 \\
b_1 & b_2 & b_3
\end{vmatrix}$$

## Transformations

### Rotation Matrix (2D)
$$R(\theta) = \begin{pmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{pmatrix}$$

### Scaling Matrix
$$S = \begin{pmatrix}
s_x & 0 & 0 \\
0 & s_y & 0 \\
0 & 0 & s_z
\end{pmatrix}$$