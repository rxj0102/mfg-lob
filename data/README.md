# data/

Data utilities for `mfg-lob`.

## Synthetic data

`synthetic_lob.py` — generate synthetic LOB snapshots for testing and
calibration.

```python
from data.synthetic_lob import SyntheticLOBConfig, generate_lob_snapshot

cfg = SyntheticLOBConfig(n_levels=20, depth_shape="power_law", depth_param=0.6)
snap = generate_lob_snapshot(cfg)
```

## Real data

For empirical studies the library targets **LOBSTER** format
(https://lobsterdata.com/):

```
<timestamp>, <type>, <order_id>, <size>, <price>, <direction>
```

To load LOBSTER data, parse each snapshot CSV into a dict with keys:
`bid_prices`, `bid_volumes`, `ask_prices`, `ask_volumes`, `mid_price`.

## Data sourcing

- **LOBSTER**: high-frequency US equity LOB data (paid, academic licence available)
- **Kaiko**: crypto LOB data (free tier)
- **Binance API**: real-time crypto depth endpoint (free)
