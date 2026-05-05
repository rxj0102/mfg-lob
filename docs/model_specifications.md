# Model Specifications

## 1. AvellanedaStoikovMFG

**References:** Avellaneda & Stoikov (2008); Cardaliaguet & Lehalle (2018)

**Class:** `mfglob.models.avellaneda_stoikov.AvellanedaStoikovMFG`

### State Space

q ∈ [−Q_max, Q_max] — signed inventory (negative = short, positive = long)

### Control

(δ_a, δ_b) — ask and bid half-spreads (distance from mid-price)

### Dynamics

Order arrivals: Poisson with rate Λ(δ) = A · exp(−k · δ).
Inventory changes by ±1 per execution; in the diffusion limit the drift is:

```
b(q, α, m) = −α           (α plays the role of net order-flow drift)
```

### Hamiltonian

```
H(q, p, M, m) = 0.5 · p² − φ·q² − ψ·m(q)·q² − 0.5·σ²·M
```

### Optimal Control

```
α*(q, p, m) = −p
```

### Optimal Spreads

```
δ_a* = max(1/k − (γ σ²/2) q,  0)    (ask half-spread)
δ_b* = max(1/k + (γ σ²/2) q,  0)    (bid half-spread)
```

At zero inventory (q=0): δ_a* = δ_b* = 1/k (symmetric, base spread).

### Execution Rate

```
Λ_a(δ_a, m, q) = A · exp(−k · δ_a) · (1 + c · m(q))
```

Mean-field coupling through the density m: higher crowd density at q increases
execution probability (competition for the same orders).

### Parameters

| Name      | Symbol | Description                          | Default | Typical range |
|-----------|--------|--------------------------------------|---------|---------------|
| A         | A      | Base arrival rate                    | 1.0     | [0.5, 5.0]    |
| k         | k      | Arrival-rate decay with depth        | 1.5     | [0.5, 5.0]    |
| phi       | φ      | Inventory running-cost coefficient   | 0.01    | [0.001, 0.1]  |
| psi       | ψ      | Crowd-aversion coefficient           | 0.005   | [0, 0.05]     |
| gamma     | γ      | Risk-aversion parameter              | 0.1     | [0.01, 1.0]   |
| sigma_mid | σ      | Mid-price volatility                 | 0.3     | [0.1, 1.0]    |
| Q_max     | Q      | Maximum inventory                    | 10.0    | [1, 20]       |
| c         | c      | Mean-field execution coupling        | 0.5     | [0, 2.0]      |

### Known Analytical Results

- **Symmetric equilibrium:** at m = const (uniform density), the optimal spread
  is exactly 1/k (Avellaneda & Stoikov 2008, eq. 13).
- **Inventory skew:** the spread widens on the side of excess inventory by
  γ σ² |q| / 2, reflecting market risk from unwinding.
- **Indifference price:** s* = s_mid − γ σ² q T  (linear in inventory).

---

## 2. OptimalExecutionMFG

**References:** Cardaliaguet & Lehalle (2018); Almgren & Chriss (2001)

**Class:** `mfglob.models.optimal_execution.OptimalExecutionMFG`

### State Space

q ∈ [0, Q₀] — remaining inventory to liquidate

### Control

ν ≥ 0 — execution rate (inventory sold per unit time)

### Dynamics

```
dq_t = −ν_t dt    (deterministic in the base model; add σ dW_t for diffusion)
```

### Running Cost

```
f(q, ν, m) = φ·q² + ψ·ν² + η·ν·M(t)
```

where M(t) = ∫ ν(t,q) m(t,q) dq is the **aggregate execution rate** (mean-field
coupling via temporary price impact coefficient η).

### Optimal Control

```
ν*(q, p, m) = max( (p − λ·η·M) / (2ψ),  0 )
```

where M is computed self-consistently from the population.

### Parameters

| Name  | Symbol | Description                         | Default | Typical range |
|-------|--------|-------------------------------------|---------|---------------|
| phi   | φ      | Inventory quadratic running cost    | 0.001   | [0.0001, 0.1] |
| psi   | ψ      | Execution cost (quadratic)          | 0.01    | [0.001, 1.0]  |
| eta   | η      | Temporary price impact coefficient  | 0.1     | [0.01, 1.0]   |
| lam   | λ      | Mean-field coupling weight          | 1.0     | [0, 2.0]      |
| sigma | σ      | Inventory diffusion (stochastic)    | 0.0     | [0, 0.2]      |
| Q0    | Q₀     | Initial inventory                   | 1.0     | [0.1, 10.0]   |

### Known Analytical Results

**Almgren-Chriss limit (η = 0, σ = 0):**

```
q*(t) = Q₀ · sinh(κ(T−t)) / sinh(κT),    κ = √(φ/ψ)
```

This is the closed-form optimal liquidation schedule for a single agent.

**MFG equilibrium:** The self-consistency condition M = ∫ ν* m dq couples every
agent's strategy.  As η increases, the equilibrium execution slows down early
(to avoid price impact from the crowd) and accelerates near T.

**Price impact scaling:** The permanent impact scales as ΔS ∝ η·Q for small η,
transitioning to sub-linear (square-root-like) behaviour for larger η.

---

## 3. LOBFormationMFG

**References:** Lasry & Lions (2007); Cardaliaguet & Lehalle (2018)

**Class:** `mfglob.models.lob_formation.LOBFormationMFG`

### State Space

x ∈ [0, x_max] — distance of the posted limit order from the mid-price

### Control

α — rate of order repositioning (signed; α > 0 moves order away from mid)

### Equilibrium Interpretation

The equilibrium density m*(x,t) is the **LOB depth profile**: m*(x,t) dx is the
total standing volume at price distance [x, x+dx] from the mid at time t.

### Running Cost

```
f(x, α, m) = c_far · x
           + c_near · exp(−κ x)
           + c_crowd · m(x)
           + c_control · α²
```

| Term         | Economic meaning                                  |
|--------------|---------------------------------------------------|
| `c_far · x`  | Inventory risk from posting far (unlikely fill)   |
| `c_near · e^{−κx}` | Adverse selection: near-mid orders fill against informed traders |
| `c_crowd · m` | Congestion: cost of competing with other agents |
| `c_control · α²` | Repositioning cost (effort to move an order)  |

### Hamiltonian

```
H(x, p, M, m) = −c_far·x − c_near·exp(−κx) − c_crowd·m(x)
                + p² / (4·c_control) − (σ²/2)·M
```

### Optimal Control

```
α*(x, p, m) = −p / (2·c_control)
```

### Parameters

| Name       | Symbol     | Description                             | Default | Range        |
|------------|------------|-----------------------------------------|---------|--------------|
| c_far      | c_far      | Far-field cost coefficient              | 0.1     | [0.01, 1.0]  |
| c_near     | c_near     | Adverse-selection amplitude             | 1.0     | [0.1, 5.0]   |
| kappa      | κ          | Adverse-selection decay rate            | 2.0     | [0.5, 10.0]  |
| c_crowd    | c_crowd    | Congestion penalty coefficient          | 0.5     | [0.01, 5.0]  |
| c_control  | c_control  | Repositioning cost coefficient          | 0.1     | [0.01, 1.0]  |
| sigma      | σ          | Order-position diffusion coefficient    | 0.5     | [0.1, 2.0]   |
| x_max      | x_max      | Maximum price distance (domain)         | 10.0    | [3.0, 20.0]  |

### Known Analytical Results

**Lasry-Lions monotonicity:** The congestion term `c_crowd · m(x)` satisfies the
Lasry-Lions condition with constant c_crowd, guaranteeing a unique equilibrium.

**Gibbs steady state (σ → ∞):**

```
m*(x) ∝ exp( −[c_far·x + c_near·(exp(−κx) − 1)] / (c_control·σ²) )
```

This is the stationary density when diffusion dominates all drift costs.

**Book shape:** the mode of m*(x) occurs where adverse selection (concentrated
near x=0) exactly balances the far-field cost.  Increasing c_crowd flattens the
profile (spreads agents more evenly) without changing the mode location much.

**Connection to empirical LOBs:** The exponential decay in empirical LOB depth
profiles (Bouchaud et al. 2002) corresponds to the c_near=0 limit of this model,
in which m*(x) ∝ exp(−c_far·x / (c_control·σ²)).

---

## 4. Connections to the Empirical Literature

| Model feature | Empirical observation | Reference |
|---------------|----------------------|-----------|
| Spread = 1/k at q=0 | Bid-ask spread ∝ 1/κ (depth decay) | Avellaneda & Stoikov (2008) |
| Inventory skew γσ²q/2 | Adverse price impact grows with |q| | Stoll (1978) |
| Optimal exec schedule | VWAP/TWAP algorithms | Almgren & Chriss (2001) |
| ΔS ∝ √Q | Square-root impact law | Gabaix et al. (2003) |
| m*(x) ∝ exp(−αx) | Exponential LOB depth profile | Bouchaud et al. (2002) |
| Congestion spreads agents | Fat-tailed LOB depth at high volume | Gould et al. (2013) |

---

## 5. Additional References

- Bouchaud, Mézard & Potters (2002). "Statistical properties of stock order books."  *Quantitative Finance* 2, 251–256.
- Gabaix, Gopikrishnan, Plerou & Stanley (2003). "A theory of power-law distributions in financial fluctuations."  *Nature* 423, 267–270.
- Gould, Porter, Williams, McDonald, Fenn & Howison (2013). "Limit order books."  *Quantitative Finance* 13, 1709–1742.
- Stoll (1978). "The supply of dealer services in securities markets."  *Journal of Finance* 33, 1133–1151.
