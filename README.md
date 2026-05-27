# Circle Packing NLP Optimizer

A Streamlit app for the classic circle-packing problem as a non-convex
nonlinear program (Pyomo + pounce). Pack $N$ circles with given radii into
the smallest enclosing rectangle. The in-app **📐 Formulation** tab walks
through the non-convex math, the interior-point solution method, and the
references — see [References](#references) below.

**Live demo:** https://circle-packing.griffith-pse.com  
**Home:** https://griffith-pse.com

## Run locally

    pip install -r requirements.txt
    streamlit run app.py

The solver is **pounce** — a primal-dual interior-point NLP solver from John
Kitchin, distributed via the
[`pyomo-pounce`](https://pypi.org/project/pyomo-pounce/) wheel, which bundles
the solver binary. No separate solver install needed; `pip install` takes
care of everything.

## Deployment

Auto-deploys to Fly.io on every push to `main` via
`.github/workflows/deploy.yml`. The `Dockerfile` builds a Python 3.12 image
and installs everything from `requirements.txt`; `fly.toml` configures
auto-stop machines. Custom domain wired through Cloudflare DNS.

- **Machine**: `shared-cpu-1x` · 1 GB RAM · single region (`ord`) · `min_machines_running=0` (auto-stops on idle).
- **Cost ceiling**: ~$3.89/mo if traffic kept the VM awake 24/7. Realistic on idle-heavy demo traffic: well under $1/mo per app. Bandwidth is effectively free under Fly's 100 GB/mo egress allowance.

## Files

- `app.py` — Streamlit UI, Pyomo model, pounce wrapper
- `Circle packing.ipynb` — formulation in a notebook
- `requirements.txt` — Python deps
- `favicon.png` — Griffith PSE blackletter G favicon
- `Dockerfile`, `fly.toml`, `.dockerignore` — Fly.io production image config
- `.github/workflows/deploy.yml` — auto-deploy pipeline

## References

[1] L. T. Biegler, *Nonlinear Programming: Concepts, Algorithms, and
Applications to Chemical Processes*. Philadelphia, PA: SIAM, 2010.
[SIAM](https://epubs.siam.org/doi/book/10.1137/1.9780898719383)

[2] M. L. Bynum, G. A. Hackebeil, W. E. Hart, C. D. Laird, B. L. Nicholson,
J. D. Siirola, J.-P. Watson, and D. L. Woodruff, *Pyomo — Optimization
Modeling in Python*, 3rd ed. Cham: Springer, 2021.
[Springer](https://link.springer.com/book/10.1007/978-3-030-68928-5)
