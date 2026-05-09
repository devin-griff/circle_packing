# Circle Packing

Non-convex NLP for the smallest enclosing rectangle

**Live demo:** https://circle-packing.griffith-pse.com  
**Home:** https://griffith-pse.com

## Run locally

    pip install -r requirements.txt
    streamlit run app.py

The solver is **ripopt** — a Rust reimplementation of IPOPT, distributed via
the [`pyomo-ripopt`](https://pypi.org/project/pyomo-ripopt/) wheel, which
bundles the solver binary. No separate solver install needed; `pip install`
takes care of everything.

## Deployment

Auto-deploys to Fly.io on every push to `main` via
`.github/workflows/deploy.yml`. The `Dockerfile` builds a Python 3.12 image
and installs everything from `requirements.txt`; `fly.toml` configures
auto-stop machines. Custom domain wired through Cloudflare DNS.

- **Machine**: `shared-cpu-1x` · 1 GB RAM · single region (`ord`) · `min_machines_running=0` (auto-stops on idle).
- **Cost ceiling**: ~$3.89/mo if traffic kept the VM awake 24/7.[^fly-pricing] Realistic on idle-heavy demo traffic: well under $1/mo per app. Bandwidth is effectively free under Fly's 100 GB/mo egress allowance.

[^fly-pricing]: Fly.io pricing as of 2026-05; published rates may shift. See https://fly.io/docs/about/pricing/.

## Files

- `app.py` — Streamlit UI and computation
- `requirements.txt` — Python deps
- `favicon.png` — Griffith PSE blackletter G favicon
- `Dockerfile`, `fly.toml`, `.dockerignore` — Fly.io production image config
- `.github/workflows/deploy.yml` — auto-deploy pipeline
