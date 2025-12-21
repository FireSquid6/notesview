# Chemistry Reference

## Atomic Structure

### Quantum Numbers

1. **Principal quantum number** ($n$): Energy level, $n = 1, 2, 3, ...$
2. **Azimuthal quantum number** ($l$): Orbital shape, $l = 0, 1, 2, ..., n-1$
3. **Magnetic quantum number** ($m_l$): Orbital orientation, $m_l = -l, ..., 0, ..., +l$
4. **Spin quantum number** ($m_s$): Electron spin, $m_s = \pm\frac{1}{2}$

### Electronic Configuration

**Aufbau Principle**: Electrons fill orbitals in order of increasing energy.

**Pauli Exclusion Principle**: No two electrons can have identical quantum numbers.

**Hund's Rule**: Electrons occupy orbitals singly before pairing up.

**Order of filling**: 1s < 2s < 2p < 3s < 3p < 4s < 3d < 4p < 5s < 4d < 5p < ...

## Chemical Bonding

### Ionic Bonding

**Lattice Energy**: $U = k \frac{Q_1 Q_2}{r_0}$

Where:
- $Q_1, Q_2$ = charges on ions
- $r_0$ = distance between ion centers
- $k$ = proportionality constant

### Covalent Bonding

**Bond Order**: $\text{Bond Order} = \frac{1}{2}(\text{bonding electrons} - \text{antibonding electrons})$

**VSEPR Theory**: Electron pairs arrange to minimize repulsion.

| Electron Pairs | Molecular Geometry |
|----------------|-------------------|
| 2 | Linear |
| 3 | Trigonal planar |
| 4 | Tetrahedral |
| 5 | Trigonal bipyramidal |
| 6 | Octahedral |

### Hybridization

- **sp³**: Tetrahedral (109.5°)
- **sp²**: Trigonal planar (120°)
- **sp**: Linear (180°)

## Thermochemistry

### Enthalpy

**Enthalpy of Formation**: $\Delta H_f^\circ$

**Hess's Law**: $\Delta H_{reaction} = \sum \Delta H_f^\circ(\text{products}) - \sum \Delta H_f^\circ(\text{reactants})$

**Heat Capacity**: $q = nC_p\Delta T$

### Entropy

**Second Law of Thermodynamics**: $\Delta S_{universe} \geq 0$

**Gibbs Free Energy**: $\Delta G = \Delta H - T\Delta S$

**Spontaneity**:
- $\Delta G < 0$: Spontaneous
- $\Delta G = 0$: Equilibrium
- $\Delta G > 0$: Non-spontaneous

## Chemical Kinetics

### Rate Laws

**Rate of Reaction**: $\text{Rate} = -\frac{1}{a}\frac{d[A]}{dt} = \frac{1}{b}\frac{d[B]}{dt}$

**Rate Law**: $\text{Rate} = k[A]^m[B]^n$

Where:
- $k$ = rate constant
- $m, n$ = reaction orders

### Integrated Rate Laws

**Zero Order**: $[A] = [A]_0 - kt$

**First Order**: $\ln[A] = \ln[A]_0 - kt$

**Second Order**: $\frac{1}{[A]} = \frac{1}{[A]_0} + kt$

### Arrhenius Equation

$$k = A e^{-E_a/RT}$$

Where:
- $A$ = pre-exponential factor
- $E_a$ = activation energy
- $R$ = gas constant
- $T$ = temperature

## Chemical Equilibrium

### Equilibrium Constant

**For reaction**: $aA + bB \rightleftharpoons cC + dD$

$$K = \frac{[C]^c[D]^d}{[A]^a[B]^b}$$

### Le Chatelier's Principle

When a system at equilibrium is disturbed, it shifts to counteract the disturbance:
- **Concentration**: Adding reactant shifts right; adding product shifts left
- **Temperature**: Increasing temperature favors endothermic direction
- **Pressure**: Increasing pressure favors side with fewer gas molecules

### Relationship between K and ΔG

$$\Delta G^\circ = -RT \ln K$$

## Acids and Bases

### pH and pOH

$$\text{pH} = -\log[H^+]$$
$$\text{pOH} = -\log[OH^-]$$
$$\text{pH} + \text{pOH} = 14 \text{ (at 25°C)}$$

### Henderson-Hasselbalch Equation

$$\text{pH} = \text{p}K_a + \log\frac{[A^-]}{[HA]}$$

### Buffer Systems

**Buffer Capacity**: Maximum amount of acid/base that can be added before significant pH change.

**Common Buffers**:
- Acetate: CH₃COOH/CH₃COO⁻ (pKₐ = 4.74)
- Phosphate: H₂PO₄⁻/HPO₄²⁻ (pKₐ = 7.21)
- Ammonia: NH₄⁺/NH₃ (pKₐ = 9.25)

## Electrochemistry

### Standard Electrode Potentials

**Cell Potential**: $E^\circ_{cell} = E^\circ_{cathode} - E^\circ_{anode}$

**Nernst Equation**: $E = E^\circ - \frac{RT}{nF}\ln Q$

At 25°C: $E = E^\circ - \frac{0.0592}{n}\log Q$

### Faraday's Laws

**First Law**: $m = \frac{ItM}{nF}$

Where:
- $m$ = mass of substance
- $I$ = current
- $t$ = time
- $M$ = molar mass
- $n$ = number of electrons
- $F$ = Faraday constant (96,485 C/mol)

### Galvanic vs Electrolytic Cells

| Property | Galvanic | Electrolytic |
|----------|----------|--------------|
| Energy | Chemical → Electrical | Electrical → Chemical |
| ΔG | Negative | Positive |
| Ecell | Positive | Negative |
| Anode | Negative | Positive |
| Cathode | Positive | Negative |

## Organic Chemistry

### Functional Groups

| Group | Structure | Example |
|-------|-----------|---------|
| Alcohol | R-OH | CH₃CH₂OH |
| Aldehyde | R-CHO | CH₃CHO |
| Ketone | R-CO-R' | CH₃COCH₃ |
| Carboxylic Acid | R-COOH | CH₃COOH |
| Ester | R-COO-R' | CH₃COOCH₃ |
| Amine | R-NH₂ | CH₃NH₂ |

### Reaction Mechanisms

**SN1 Mechanism**: 
1. Ionization: R-X → R⁺ + X⁻ (slow)
2. Nucleophilic attack: R⁺ + Nu⁻ → R-Nu (fast)

**SN2 Mechanism**: Nu⁻ + R-X → Nu-R + X⁻ (concerted)

**E1 Mechanism**: 
1. Ionization: R-X → R⁺ + X⁻ (slow)
2. Elimination: R⁺ → alkene + H⁺ (fast)

**E2 Mechanism**: Base + R-X → alkene + Base-H⁺ + X⁻ (concerted)

## Spectroscopy

### IR Spectroscopy

**Beer-Lambert Law**: $A = \epsilon bc$

Where:
- $A$ = absorbance
- $\epsilon$ = molar absorptivity
- $b$ = path length
- $c$ = concentration

**Common IR Frequencies**:
- O-H stretch: 3200-3600 cm⁻¹
- C-H stretch: 2800-3000 cm⁻¹
- C=O stretch: 1650-1750 cm⁻¹
- C=C stretch: 1620-1680 cm⁻¹

### NMR Spectroscopy

**Chemical Shift**: $\delta = \frac{\nu_{sample} - \nu_{reference}}{\nu_{spectrometer}} \times 10^6$ ppm

**Coupling Constant**: $J$ (Hz) - distance between split peaks

**Integration**: Relative area under peaks ∝ number of protons

## Periodic Trends

### Atomic Properties

**Atomic Radius**: Decreases across period, increases down group

**Ionization Energy**: $M(g) \rightarrow M^+(g) + e^-$
- Increases across period, decreases down group

**Electron Affinity**: $M(g) + e^- \rightarrow M^-(g)$
- Generally increases across period (becomes more negative)

**Electronegativity**: Increases across period, decreases down group

### Chemical Properties

**Metallic Character**: Decreases across period, increases down group

**Oxidizing Power**: Increases across period (for main group elements)

**Reducing Power**: Decreases across period, increases down group