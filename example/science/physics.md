# Physics Equations and Concepts

## Classical Mechanics

### Newton's Laws
1. **First Law**: An object at rest stays at rest, and an object in motion stays in motion, unless acted upon by a net external force.

2. **Second Law**: $\vec{F} = m\vec{a}$

3. **Third Law**: For every action, there is an equal and opposite reaction.

### Kinematics

**Position**: $x(t) = x_0 + v_0t + \frac{1}{2}at^2$

**Velocity**: $v(t) = v_0 + at$

**Acceleration**: $a = \frac{dv}{dt} = \frac{d^2x}{dt^2}$

### Energy

**Kinetic Energy**: $KE = \frac{1}{2}mv^2$

**Potential Energy**: $PE = mgh$ (gravitational)

**Conservation of Energy**: $E_{total} = KE + PE = \text{constant}$

### Rotational Motion

**Angular velocity**: $\omega = \frac{d\theta}{dt}$

**Angular acceleration**: $\alpha = \frac{d\omega}{dt}$

**Moment of inertia**: $I = \sum_i m_i r_i^2$

**Rotational kinetic energy**: $KE_{rot} = \frac{1}{2}I\omega^2$

**Torque**: $\vec{\tau} = \vec{r} \times \vec{F} = I\vec{\alpha}$

## Thermodynamics

### Laws of Thermodynamics

**Zeroth Law**: If two systems are in thermal equilibrium with a third, they are in thermal equilibrium with each other.

**First Law**: $\Delta U = Q - W$
Where:
- $\Delta U$ = change in internal energy
- $Q$ = heat added to system
- $W$ = work done by system

**Second Law**: The entropy of an isolated system never decreases:
$$\Delta S \geq 0$$

**Third Law**: The entropy of a perfect crystal approaches zero as temperature approaches absolute zero.

### Gas Laws

**Ideal Gas Law**: $PV = nRT$

**Van der Waals Equation**: $\left(P + \frac{a}{V^2}\right)(V - b) = RT$

### Heat Transfer

**Conduction**: $\dot{Q} = -kA\frac{dT}{dx}$ (Fourier's Law)

**Convection**: $\dot{Q} = hA(T_s - T_\infty)$ (Newton's Law of Cooling)

**Radiation**: $\dot{Q} = \sigma A T^4$ (Stefan-Boltzmann Law)

## Electromagnetism

### Electric Fields

**Coulomb's Law**: $\vec{F} = k\frac{q_1 q_2}{r^2}\hat{r}$

**Electric Field**: $\vec{E} = \frac{\vec{F}}{q}$

**Electric Potential**: $V = \frac{U}{q}$

**Gauss's Law**: $\oint \vec{E} \cdot d\vec{A} = \frac{Q_{enc}}{\epsilon_0}$

### Magnetic Fields

**Lorentz Force**: $\vec{F} = q(\vec{E} + \vec{v} \times \vec{B})$

**Biot-Savart Law**: $d\vec{B} = \frac{\mu_0}{4\pi} \frac{I d\vec{l} \times \hat{r}}{r^2}$

**Ampère's Law**: $\oint \vec{B} \cdot d\vec{l} = \mu_0 I_{enc}$

### Maxwell's Equations

In vacuum:

$$\nabla \cdot \vec{E} = \frac{\rho}{\epsilon_0}$$ (Gauss's Law)

$$\nabla \cdot \vec{B} = 0$$ (No magnetic monopoles)

$$\nabla \times \vec{E} = -\frac{\partial \vec{B}}{\partial t}$$ (Faraday's Law)

$$\nabla \times \vec{B} = \mu_0 \vec{J} + \mu_0 \epsilon_0 \frac{\partial \vec{E}}{\partial t}$$ (Ampère-Maxwell Law)

## Quantum Mechanics

### Fundamental Principles

**De Broglie Wavelength**: $\lambda = \frac{h}{p}$

**Heisenberg Uncertainty Principle**: $\Delta x \Delta p \geq \frac{\hbar}{2}$

**Schrödinger Equation** (time-dependent):
$$i\hbar \frac{\partial \Psi}{\partial t} = \hat{H}\Psi$$

**Schrödinger Equation** (time-independent):
$$\hat{H}\psi = E\psi$$

### Wave Function

**Normalization**: $\int_{-\infty}^{\infty} |\Psi(x,t)|^2 dx = 1$

**Probability Density**: $\rho(x,t) = |\Psi(x,t)|^2$

**Expectation Value**: $\langle \hat{A} \rangle = \int_{-\infty}^{\infty} \Psi^* \hat{A} \Psi \, dx$

### Quantum Harmonic Oscillator

**Energy Levels**: $E_n = \hbar \omega \left(n + \frac{1}{2}\right)$ where $n = 0, 1, 2, ...$

**Ground State Wave Function**: $\psi_0(x) = \left(\frac{m\omega}{\pi\hbar}\right)^{1/4} e^{-\frac{m\omega x^2}{2\hbar}}$

### Hydrogen Atom

**Energy Levels**: $E_n = -\frac{13.6 \text{ eV}}{n^2}$ where $n = 1, 2, 3, ...$

**Bohr Radius**: $a_0 = \frac{\hbar^2}{me^2} = 0.529 \times 10^{-10}$ m

## Relativity

### Special Relativity

**Time Dilation**: $\Delta t = \gamma \Delta t_0$ where $\gamma = \frac{1}{\sqrt{1 - \frac{v^2}{c^2}}}$

**Length Contraction**: $L = \frac{L_0}{\gamma}$

**Mass-Energy Equivalence**: $E = mc^2$

**Energy-Momentum Relation**: $E^2 = (pc)^2 + (mc^2)^2$

**Lorentz Transformation**:
$$x' = \gamma(x - vt)$$
$$t' = \gamma\left(t - \frac{vx}{c^2}\right)$$

### General Relativity

**Einstein Field Equations**: $G_{\mu\nu} = \frac{8\pi G}{c^4} T_{\mu\nu}$

**Schwarzschild Metric** (spherically symmetric):
$$ds^2 = -\left(1 - \frac{2GM}{c^2 r}\right)c^2 dt^2 + \left(1 - \frac{2GM}{c^2 r}\right)^{-1}dr^2 + r^2 d\theta^2 + r^2 \sin^2\theta \, d\phi^2$$

**Schwarzschild Radius**: $r_s = \frac{2GM}{c^2}$

## Wave Phenomena

### Wave Equation

**General Form**: $\frac{\partial^2 \psi}{\partial t^2} = v^2 \nabla^2 \psi$

**1D Solution**: $\psi(x,t) = A\sin(kx - \omega t + \phi)$

**Wave Speed**: $v = \frac{\omega}{k} = f\lambda$

### Interference

**Constructive**: Path difference $= n\lambda$ where $n = 0, 1, 2, ...$

**Destructive**: Path difference $= (n + \frac{1}{2})\lambda$ where $n = 0, 1, 2, ...$

### Doppler Effect

**Source moving**: $f' = f \frac{v \pm v_r}{v \pm v_s}$

Where:
- $f'$ = observed frequency
- $f$ = source frequency  
- $v$ = wave speed
- $v_r$ = receiver velocity
- $v_s$ = source velocity

## Constants

| Constant | Symbol | Value |
|----------|--------|-------|
| Speed of light | $c$ | $2.998 \times 10^8$ m/s |
| Planck constant | $h$ | $6.626 \times 10^{-34}$ J⋅s |
| Reduced Planck constant | $\hbar$ | $1.055 \times 10^{-34}$ J⋅s |
| Gravitational constant | $G$ | $6.674 \times 10^{-11}$ N⋅m²/kg² |
| Elementary charge | $e$ | $1.602 \times 10^{-19}$ C |
| Electron mass | $m_e$ | $9.109 \times 10^{-31}$ kg |
| Proton mass | $m_p$ | $1.673 \times 10^{-27}$ kg |
| Avogadro's number | $N_A$ | $6.022 \times 10^{23}$ mol⁻¹ |
| Boltzmann constant | $k_B$ | $1.381 \times 10^{-23}$ J/K |