# Calculus Fundamentals

## Derivatives

The derivative of a function $f(x)$ at point $x$ is defined as:
$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

### Common Derivatives

| Function | Derivative |
|----------|------------|
| $x^n$ | $nx^{n-1}$ |
| $e^x$ | $e^x$ |
| $\ln(x)$ | $\frac{1}{x}$ |
| $\sin(x)$ | $\cos(x)$ |
| $\cos(x)$ | $-\sin(x)$ |

### Product Rule
$$(fg)' = f'g + fg'$$

### Chain Rule
$$\frac{d}{dx}[f(g(x))] = f'(g(x)) \cdot g'(x)$$

## Integrals

The fundamental theorem of calculus:
$$\int_a^b f'(x) \, dx = f(b) - f(a)$$

### Integration by Parts
$$\int u \, dv = uv - \int v \, du$$

## Applications

### Area Under a Curve
The area between $f(x)$ and the x-axis from $a$ to $b$:
$$A = \int_a^b f(x) \, dx$$

### Volume of Revolution
Rotating $f(x)$ around the x-axis:
$$V = \pi \int_a^b [f(x)]^2 \, dx$$