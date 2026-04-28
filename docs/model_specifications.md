# Model Specifications

## 1. AvellanedaStoikovMFG

**Reference:** Avellaneda & Stoikov (2008); Cardaliaguet & Lehalle (2018)

### State space
q ∈ [q_min, q_max] — signed inventory (negative = short, positive = long)

### Control
(δ_bid, δ_ask) — bid and ask half-spreads (distance from mid-price)

### Dynamics
Order arrivals: Poisson with rate Λ(δ) = Λ₀ exp(-κδ).
Inventory changes by ±1 per execution.

### Hamiltonian
```
H(q, p, M, m) = (Λ₀/κ) exp(-1) [exp(-κ·p) + exp(κ·p)]
```
(symmetric bid/ask structure; see Proposition 3.1 in Cardaliaguet-Lehalle)

### Parameters
| Name    | Symbol | Description                     | Typical value |
|---------|--------|---------------------------------|---------------|
| sigma   | σ      | Mid-price volatility            | 0.3           |
| gamma   | γ      | Inventory risk aversion         | 0.1           |
| Lambda0 | Λ₀     | Base order arrival rate         | 1.0           |
| kappa   | κ      | Arrival rate decay with depth   | 1.5           |
| A       | A      | Terminal inventory penalty      | 0.5           |

---

## 2. OptimalExecutionMFG

**Reference:** Cardaliaguet & Lehalle (2018); Almgren & Chriss (2001)

### State space
q ∈ [0, Q] — remaining inventory to execute

### Control
v ≥ 0 — trading speed (inventory consumption rate)

### Dynamics
dq_t = -v_t dt

### Running cost
f(q, v, m) = ε·v² + η·v·M(t) + φ·q²

where M(t) = ∫ v(q,t) m(q,t) dq is the aggregate market order flow
(mean-field coupling).

---

## 3. LOBFormationMFG

**Reference:** Lasry & Lions (2007)

### State space
x ∈ [0, x_max] — distance of limit order from mid-price

### Control
a — rate of order relocation

### Equilibrium interpretation
The equilibrium density m(x,t) is the LOB depth profile: m(x,t)dx is
the standing volume at price distance x at time t.

### Mean-field coupling
Congestion cost: f(x,a,m) = φ·x² + α·m(x,t).

Satisfies Lasry-Lions monotonicity → unique equilibrium.
