# =============================================================================
# Circle Packing — a Streamlit tutorial app.
#
# Pack N non-overlapping circles into the smallest enclosing rectangle. This is
# the textbook non-convex NLP from Biegler's "Nonlinear Programming" — the
# pairwise non-overlap inequalities are non-convex, so multiple local optima
# are expected. The "Re-randomize and solve" button is the user's tool for
# exploring them.
#
# Mathematical formulation (see the Formulation tab for full LaTeX):
#   minimize   W * H
#   s.t.       r_i <= x_i <= W - r_i,   forall i
#              r_i <= y_i <= H - r_i,   forall i
#              (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2,  forall i < j
#              W, H >= 0
#
# Library roadmap:
#   - streamlit  — UI framework. Each interaction reruns this script
#                  top-to-bottom; persistent values live in `st.session_state`.
#   - pyomo      — algebraic modeling: sets, params, vars, constraints,
#                  objective. Continuous variables only.
#   - rIPOPT    — the NLP solver, a Rust reimplementation of IPOPT
#                  (primal-dual interior-point). Called as a subprocess via
#                  Pyomo. Binary ships in the `pyomo-ripopt` wheel.
#   - matplotlib — circle plotting (Altair makes circle aspect ratios painful).
#
# File roadmap:
#   1. Page config + CSS.
#   2. Sidebar inputs (set-then-solve flow, like quad-tank).
#   3. Defaults / state helpers.
#   4. solve_model      — builds and solves the Pyomo NLP.
#   5. build_packing_fig — matplotlib figure of rectangle + circles.
#   6. Tab renderers     — Optimizer / Data / Formulation / Logs.
#   7. Main layout       — auto-solve on first load, then four tabs.
# =============================================================================

import base64
import io
import contextlib
import math
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pyomo.environ as pyo
# Registers `ripopt` with pyo.SolverFactory via decorator side-effect; the
# wheel also bundles the solver binary so no system install is required.
import pyomo_ripopt  # noqa: F401
from pyomo.common.errors import ApplicationError


# `set_page_config` must be the first Streamlit call. Wide layout + open
# sidebar gives the packing plot enough horizontal room.
st.set_page_config(
    page_title="Circle Packing",
    page_icon="favicon.png",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── CSS ──────────────────────────────────────────────────────────────────────
# Sidebar text gets `user-select: none` so dragging widgets doesn't
# accidentally select labels. Top padding leaves room for the corner-pinned
# home logo while keeping the title close to the page top.
st.markdown("""
<style>
section[data-testid="stSidebar"] {
    user-select: none;
    -webkit-user-select: none;
}
.home-logo-corner {
    position: fixed;
    top: 0.5rem;
    left: 0.75rem;
    z-index: 999999;
}
.home-logo-corner img {
    width: 32px;
    height: 32px;
    border-radius: 4px;
    display: block;
}
.block-container,
[data-testid="stMainBlockContainer"] {
    padding-top: 4rem !important;
    padding-bottom: 0rem !important;
}
</style>
""", unsafe_allow_html=True)


# Home link: clicking the Griffith PSE logo navigates back to the portfolio
# site. Same-tab navigation since the user is leaving the demo. Lives at the
# top of the sidebar (the upper-left of the page when expanded), matching the
# quad-tank pattern. Image is embedded from the local favicon.png as a base64
# data URL — the link still navigates to griffith-pse.com when clicked, but
# loading the page itself doesn't make any third-party request.
_FAVICON_DATA_URL = "data:image/png;base64," + base64.b64encode(
    (Path(__file__).parent / "favicon.png").read_bytes()
).decode()
st.sidebar.markdown(
    f'<a class="home-logo-corner" href="https://griffith-pse.com" target="_self">'
    f'<img src="{_FAVICON_DATA_URL}" '
        f'alt="Griffith PSE — home" />'
    f'</a>',
    unsafe_allow_html=True,
)


# ── Constants / defaults ─────────────────────────────────────────────────────
MIN_N = 2
MAX_N = 30
DEFAULT_N = 10
DEFAULT_RADIUS = 1.0


def default_data(n=DEFAULT_N):
    """Build the default scenario: n unit circles with placeholder initial
    positions. Initial (x0, y0) values are seeded with a deterministic
    grid-ish layout so the data table looks tidy on first render; the
    solver overwrites them once a Random initial guess is generated."""
    radii = [DEFAULT_RADIUS] * n
    # Light grid placement just so the editor isn't full of zeros.
    cols = int(math.ceil(math.sqrt(n)))
    pts = []
    for i in range(n):
        cx = (i % cols) * 2.2 * DEFAULT_RADIUS + DEFAULT_RADIUS
        cy = (i // cols) * 2.2 * DEFAULT_RADIUS + DEFAULT_RADIUS
        pts.append((cx, cy))
    return {
        "radii": radii,
        "x0": [p[0] for p in pts],
        "y0": [p[1] for p in pts],
    }


def random_initial_positions(radii, seed=None, box_scale=1.0):
    """Generate a random initial layout. Each circle's center is sampled
    uniformly inside a square of side `box_scale * sum(r)` — the side length
    is large enough to plausibly contain the optimum but small enough that
    rIPOPT doesn't wander."""
    rng = np.random.default_rng(seed)
    side = box_scale * sum(radii)
    n = len(radii)
    xs = rng.uniform(0.0, side, size=n).tolist()
    ys = rng.uniform(0.0, side, size=n).tolist()
    return xs, ys


# ── State init ───────────────────────────────────────────────────────────────
# Streamlit re-executes this script on every interaction. Anything that must
# persist between runs lives in `st.session_state`. Keys we use:
#   - data:       current problem instance (radii, x0, y0)
#   - res:        most recent solver result, or None
#   - seed:       integer seed for the Random initial-guess generator
#   - guess_mode: "Random" | "User-specified"

def init_state():
    if "data" not in st.session_state:
        st.session_state.data = default_data(DEFAULT_N)
    if "res" not in st.session_state:
        st.session_state.res = None
    if "seed" not in st.session_state:
        st.session_state.seed = 0
    if "guess_mode" not in st.session_state:
        st.session_state.guess_mode = "Random"
    # Editor revision counter — bumped whenever we change `data` from outside
    # the data_editor widget, so the widget gets a fresh key and rebuilds
    # from the new base data instead of replaying stale edits on top.
    if "editor_rev" not in st.session_state:
        st.session_state.editor_rev = 0


init_state()


# ── Sidebar ──────────────────────────────────────────────────────────────────
# Set-then-solve workflow: choose N and the initial guess strategy, then
# click Solve. Non-convex NLPs are heavier than LPs so we don't auto-solve
# on every widget change.

st.sidebar.header("Configuration")

# Number of circles. When the user changes N, we resize the radii / x0 / y0
# arrays in `data` so other tabs see the new instance immediately.
n_circles = st.sidebar.number_input(
    "Number of circles",
    min_value=MIN_N,
    max_value=MAX_N,
    value=len(st.session_state.data["radii"]),
    step=1,
    key="n_input",
)

if n_circles != len(st.session_state.data["radii"]):
    cur = st.session_state.data
    n = int(n_circles)
    if n > len(cur["radii"]):
        # Extend with default unit circles + grid-ish initial positions.
        extra = n - len(cur["radii"])
        cur["radii"].extend([DEFAULT_RADIUS] * extra)
        cols = int(math.ceil(math.sqrt(n)))
        for i in range(len(cur["x0"]), n):
            cx = (i % cols) * 2.2 * DEFAULT_RADIUS + DEFAULT_RADIUS
            cy = (i // cols) * 2.2 * DEFAULT_RADIUS + DEFAULT_RADIUS
            cur["x0"].append(cx)
            cur["y0"].append(cy)
    else:
        cur["radii"] = cur["radii"][:n]
        cur["x0"] = cur["x0"][:n]
        cur["y0"] = cur["y0"][:n]
    st.session_state.res = None  # invalidate previous solve
    st.session_state.editor_rev += 1  # force the data editor to rebuild

# Initial-guess strategy. Random samples uniformly inside a bounding box;
# User-specified uses the (x0, y0) values from the Data tab. The radio's
# persisted state lives under a dedicated widget key so we can keep our
# own `guess_mode` mirror around (used by _gather_initial_guess) without
# Streamlit complaining about mutating a widget-backed key in the same run.
if "guess_mode_radio" not in st.session_state:
    st.session_state["guess_mode_radio"] = st.session_state.guess_mode
guess_mode = st.sidebar.radio(
    "Initial guess",
    ["Random", "User-specified"],
    key="guess_mode_radio",
    help="Random samples new starting positions inside a bounding box. "
         "User-specified uses the (x0, y0) values from the Data tab.",
)
st.session_state.guess_mode = guess_mode

# Seed control — only meaningful in Random mode. A separate seed input lets
# the user reproduce a particular local optimum if they hit something
# interesting. The "Re-randomize and solve" button bumps the seed before
# the widget runs (via session_state.seed_input), so the widget redisplays
# the new value on the next render.
if guess_mode == "Random":
    if "seed_input" not in st.session_state:
        st.session_state["seed_input"] = int(st.session_state.seed)
    seed_val = st.sidebar.number_input(
        "Random seed",
        min_value=0,
        max_value=1_000_000,
        step=1,
        key="seed_input",
        help="Bump this (or click Re-randomize and solve) for a new initial layout.",
    )
    st.session_state.seed = int(seed_val)

# Solve buttons. The second button combines "new random init" with "solve" —
# the canonical exploration path for a non-convex problem. The button uses
# `on_click` so the seed bump runs *before* the widget is instantiated on
# the next rerun, which is the only safe way to mutate a widget-backed key
# (`seed_input`) without Streamlit raising.
def _bump_seed_and_request_solve():
    st.session_state.seed = int(st.session_state.get("seed", 0)) + 1
    if "seed_input" in st.session_state:
        st.session_state["seed_input"] = st.session_state.seed
    # Force Random mode for this solve via the widget key directly. Setting
    # the radio's backing key here (in a callback, before the widget runs
    # again on the next rerun) is allowed.
    st.session_state["guess_mode_radio"] = "Random"
    st.session_state.guess_mode = "Random"
    st.session_state["_pending_solve"] = True

solve_btn = st.sidebar.button(
    "Solve",
    type="primary",
    width="stretch",
    key="solve_btn",
)
st.sidebar.button(
    "Re-randomize and solve",
    width="stretch",
    key="resolve_btn",
    on_click=_bump_seed_and_request_solve,
    help="Generate a new random initial layout and solve. Use this to "
         "explore different local optima.",
)


# ── Solver ───────────────────────────────────────────────────────────────────
#
# `solve_model` builds and solves the circle-packing NLP. The model is
# straightforward Pyomo — sets, vars, bounds, the bilinear W*H objective,
# and the pairwise non-overlap constraints.

def solve_model(radii, x0, y0):
    """Build and solve the circle-packing NLP. Returns a result dict
    consumable by the UI layer; on success it includes the optimized W, H,
    and circle centers; on failure a status string + the captured log."""
    n = len(radii)
    if n < 2:
        return {
            "status": "no_circles",
            "log": "",
            "W": None, "H": None,
            "x": [], "y": [], "radii": list(radii),
        }

    m = pyo.ConcreteModel()
    m.I = pyo.Set(initialize=list(range(n)))

    # Per-circle radius parameter.
    m.r = pyo.Param(m.I, initialize={i: float(radii[i]) for i in range(n)})

    # An upper bound on W and H keeps rIPOPT well-conditioned. 2 * sum(r)
    # is generous: any reasonable packing fits inside a square of that size
    # since side >= max(2*r_i) is the trivial lower bound.
    sum_r = float(sum(radii))
    big = max(2.0 * sum_r, 4.0 * max(radii))

    # Decision variables. W, H are the rectangle dimensions; x_i, y_i are
    # circle centers. All non-negative; centers also have radius-dependent
    # bounds via the inside-rectangle constraints below.
    m.W = pyo.Var(bounds=(0.0, big), initialize=2.0 * sum_r)
    m.H = pyo.Var(bounds=(0.0, big), initialize=2.0 * sum_r)
    m.x = pyo.Var(m.I, bounds=(0.0, big))
    m.y = pyo.Var(m.I, bounds=(0.0, big))

    # Initial guess. The solver needs a starting point in the interior of
    # the feasible region (or close to it) — we let the user pick.
    for i in range(n):
        m.x[i].value = float(x0[i]) if i < len(x0) else float(radii[i])
        m.y[i].value = float(y0[i]) if i < len(y0) else float(radii[i])

    # Inside-rectangle constraints: each circle must fit horizontally and
    # vertically. Equivalent to r_i <= x_i <= W - r_i.
    def x_upper(m, i):
        return m.x[i] + m.r[i] <= m.W
    def x_lower(m, i):
        return m.x[i] >= m.r[i]
    def y_upper(m, i):
        return m.y[i] + m.r[i] <= m.H
    def y_lower(m, i):
        return m.y[i] >= m.r[i]
    m.x_upper = pyo.Constraint(m.I, rule=x_upper)
    m.x_lower = pyo.Constraint(m.I, rule=x_lower)
    m.y_upper = pyo.Constraint(m.I, rule=y_upper)
    m.y_lower = pyo.Constraint(m.I, rule=y_lower)

    # Pairwise non-overlap. Squared form so the constraint stays smooth at
    # the boundary (sqrt would have a non-differentiable point at distance
    # zero). This is the non-convex piece of the model.
    def no_overlap(m, i, j):
        if i >= j:
            return pyo.Constraint.Skip
        return (m.x[i] - m.x[j])**2 + (m.y[i] - m.y[j])**2 \
               >= (m.r[i] + m.r[j])**2
    m.no_overlap = pyo.Constraint(m.I, m.I, rule=no_overlap)

    # Objective: minimize area. Bilinear W*H — also non-convex, but rIPOPT
    # handles it fine.
    m.area = pyo.Objective(expr=m.W * m.H, sense=pyo.minimize)

    # Solve. rIPOPT's binary is bundled in pyomo-ripopt so SolverFactory
    # finds it without any path lookup. `tee=True` streams solver output
    # to stdout, which we redirect into a StringIO so it can land in the
    # Logs tab.
    buf = io.StringIO()
    solver = pyo.SolverFactory("ripopt")
    try:
        with contextlib.redirect_stdout(buf):
            results = solver.solve(m, tee=True)
    except ApplicationError as e:
        return {
            "status": "solver_missing",
            "message": (
                "rIPOPT solver binary not found. The `pyomo-ripopt` wheel "
                "should bundle it — check your environment. "
                f"({e})"
            ),
            "log": buf.getvalue(),
            "W": None, "H": None,
            "x": [], "y": [], "radii": list(radii),
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Solver error: {e}",
            "log": buf.getvalue(),
            "W": None, "H": None,
            "x": [], "y": [], "radii": list(radii),
        }

    tc = results.solver.termination_condition
    status = str(tc)

    # Extract numeric values. Even on a non-optimal status (locallyOptimal,
    # maxIterations, etc.) we usually still have a usable layout, so always
    # pull the values out and let the UI label the status.
    try:
        W_val = float(pyo.value(m.W))
        H_val = float(pyo.value(m.H))
        xs = [float(pyo.value(m.x[i])) for i in range(n)]
        ys = [float(pyo.value(m.y[i])) for i in range(n)]
        area_val = float(pyo.value(m.area))
    except Exception:
        return {
            "status": status,
            "log": buf.getvalue(),
            "W": None, "H": None,
            "x": [], "y": [], "radii": list(radii),
        }

    return {
        "status": status,
        "log": buf.getvalue(),
        "W": W_val,
        "H": H_val,
        "area": area_val,
        "x": xs,
        "y": ys,
        "radii": list(radii),
        "x0": list(x0),
        "y0": list(y0),
    }


# ── Plotting ─────────────────────────────────────────────────────────────────
#
# `build_packing_fig` returns a matplotlib Figure showing the rectangle and
# all circles inside it. Equal aspect ratio, indexed circle labels, soft
# fill colors. Used by the Optimizer tab.

def build_packing_fig(res):
    fig, ax = plt.subplots(figsize=(7, 7))
    W = res["W"]
    H = res["H"]
    xs = res["x"]
    ys = res["y"]
    rs = res["radii"]
    n = len(rs)

    # Rectangle outline. Solid black, no fill.
    rect = mpatches.Rectangle(
        (0.0, 0.0), W, H,
        linewidth=2, edgecolor="black", facecolor="none",
    )
    ax.add_patch(rect)

    # Color cycle — matplotlib's default tab10/tab20 looks fine for ~30.
    cmap = plt.get_cmap("tab20" if n > 10 else "tab10")

    for i in range(n):
        color = cmap(i % cmap.N)
        circle = mpatches.Circle(
            (xs[i], ys[i]), rs[i],
            linewidth=1.5, edgecolor="black",
            facecolor=color, alpha=0.55,
        )
        ax.add_patch(circle)
        # Center dot.
        ax.plot([xs[i]], [ys[i]], marker=".", color="black", markersize=4)
        # Index label centered on the circle.
        ax.text(
            xs[i], ys[i], str(i + 1),
            ha="center", va="center",
            fontsize=9, fontweight="bold",
            color="black",
        )

    # Padding around the rectangle so circle outlines aren't clipped.
    pad = 0.08 * max(W, H, 1.0)
    ax.set_xlim(-pad, W + pad)
    ax.set_ylim(-pad, H + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig


# ── Status helpers ───────────────────────────────────────────────────────────

def status_label(status):
    """Map raw Pyomo termination conditions onto a short user-facing label
    plus a streamlit-style severity ("ok" | "warn" | "err")."""
    s = (status or "").lower()
    if s == "optimal":
        return "Optimal", "ok"
    if "locallyoptimal" in s.replace(" ", ""):
        return "Locally optimal", "ok"
    if "feasible" in s and "infeasible" not in s:
        return "Feasible (suboptimal)", "warn"
    if "maxiterations" in s.replace(" ", "") or "iteration" in s:
        return "Max iterations reached", "warn"
    if "infeasible" in s:
        return "Infeasible", "err"
    if "unbounded" in s:
        return "Unbounded", "err"
    if s in ("solver_missing", "error", "no_circles"):
        return status, "err"
    return status, "warn"


# ── Solve dispatch ───────────────────────────────────────────────────────────
# Handles both sidebar buttons + the first-load auto-solve. We also re-pull
# initial guesses from the right source depending on the radio selection.

def _gather_initial_guess():
    """Return the (x0, y0) lists for the current solve, depending on the
    Random vs User-specified radio. Random regenerates from the current
    seed; User-specified reads the editor-backed values out of state.data."""
    radii = st.session_state.data["radii"]
    if st.session_state.guess_mode == "Random":
        x0, y0 = random_initial_positions(radii, seed=st.session_state.seed)
        # Mirror the random guess into state.data so the Data tab shows
        # what the solver actually used. Bumping editor_rev forces the
        # data editor on the Data tab to re-display the new positions.
        st.session_state.data["x0"] = x0
        st.session_state.data["y0"] = y0
        st.session_state.editor_rev += 1
        return x0, y0
    return list(st.session_state.data["x0"]), list(st.session_state.data["y0"])


def _do_solve():
    """Run a solve against the current data + initial guess and stash the
    result. Wrapped in a spinner so the UI shows progress on slower N."""
    radii = list(st.session_state.data["radii"])
    x0, y0 = _gather_initial_guess()
    with st.spinner("Running rIPOPT optimization..."):
        res = solve_model(radii, x0, y0)
    st.session_state.res = res


# Auto-solve on first load so the page isn't empty before the user interacts.
if st.session_state.res is None:
    _do_solve()
    st.rerun()


# Manual solve.
if solve_btn:
    _do_solve()
    st.rerun()


# Re-randomize-and-solve: the on_click callback bumped the seed and forced
# Random mode before this run started, then set the _pending_solve flag.
# We pop it here so the solve runs exactly once.
if st.session_state.pop("_pending_solve", False):
    _do_solve()
    st.rerun()


# ── Title block ──────────────────────────────────────────────────────────────
st.markdown(
    "<h2 style='margin: 0 0 0.25rem 0; padding: 0; font-size: 1.5rem; font-weight: 700;'>"
    "Circle Packing "
    "<span style='font-size: 1.15rem; font-weight: 400; color: #6b7280;'>"
    "powered by "
    "<a href='https://github.com/jkitchin/ripopt' target='_blank' "
    "style='color: #6b7280; text-decoration: underline;'>rIPOPT</a>"
    "</span>"
    "</h2>",
    unsafe_allow_html=True,
)
_caption_col, _ = st.columns([5, 4])
with _caption_col:
    st.markdown(
        "Pack N non-overlapping circles into the smallest enclosing rectangle. "
        "This is a classic non-convex NLP from Biegler's *Nonlinear Programming* "
        "textbook — the pairwise non-overlap inequalities are non-convex, so "
        "different starting points often land at different local optima. Configure "
        "circles in the sidebar (or **Data** tab) and click **Solve**; use "
        "**Re-randomize and solve** to explore alternate local minima."
    )


# ── Tabs ─────────────────────────────────────────────────────────────────────

tab_opt, tab_data, tab_form, tab_logs = st.tabs(
    ["🎯 Optimizer", "📋 Data", "📐 Formulation", "📜 Logs"]
)


def render_optimizer_tab():
    res = st.session_state.res
    if res is None:
        st.info("Click **Solve** in the sidebar to run the optimizer.")
        return

    # Status banner — only shown when the solver returned something other
    # than a clean (locally) optimal result.
    label, severity = status_label(res["status"])
    if severity == "err":
        msg = res.get("message") or label
        st.error(f"Solver status: {msg}")
        return
    if severity == "warn":
        st.warning(
            f"Solver status: {label} — results may be suboptimal. Try "
            "Re-randomize and solve to explore other local minima."
        )

    # Headline metrics — width, height, area.
    c1, c2, c3 = st.columns(3)
    c1.metric("Width (W)", f"{res['W']:.4f}")
    c2.metric("Height (H)", f"{res['H']:.4f}")
    c3.metric("Area (W × H)", f"{res.get('area', res['W'] * res['H']):.4f}")

    # Plot. Centered in a narrow middle column so the figure isn't stretched
    # full-width on a wide monitor.
    _, mid, _ = st.columns([1, 3, 1])
    with mid:
        fig = build_packing_fig(res)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # Status caption beneath the plot — visible even on success so the user
    # always knows what kind of optimum they're looking at.
    st.caption(f"Solver status: **{label}**  ·  rIPOPT termination: `{res['status']}`")


def render_data_tab():
    data = st.session_state.data
    n = len(data["radii"])

    st.markdown(
        "Edit per-circle radii and initial-guess positions. Use the **Sidebar** "
        "to switch between Random and User-specified initial guesses; "
        "User-specified mode passes the (x0, y0) columns below directly to "
        "the solver."
    )

    df = pd.DataFrame({
        "Index": list(range(1, n + 1)),
        "Radius": data["radii"],
        "x0": data["x0"],
        "y0": data["y0"],
    })

    # The editor key includes a revision counter that gets bumped whenever
    # `data` is modified outside the editor (Reset, Randomize positions, N
    # changed, Random-mode solve overwrites x0/y0). A fresh key forces the
    # editor to rebuild from the new base data instead of replaying stale
    # edits on top.
    edited = st.data_editor(
        df,
        num_rows="dynamic",
        width="stretch",
        column_config={
            "Index": st.column_config.NumberColumn("Index", disabled=True),
            "Radius": st.column_config.NumberColumn("Radius", min_value=0.0, step=0.1, format="%.4f"),
            "x0":     st.column_config.NumberColumn("x₀ (initial)", step=0.1, format="%.4f"),
            "y0":     st.column_config.NumberColumn("y₀ (initial)", step=0.1, format="%.4f"),
        },
        key=f"data_editor_{st.session_state.editor_rev}",
        height=min(600, (n + 2) * 35 + 3),
    )

    # Validate and reflect edits back into state.data. We coerce numerics,
    # drop blank rows, clamp negative radii to zero, and enforce MAX_N.
    df_clean = edited.copy()
    df_clean["Radius"] = pd.to_numeric(df_clean["Radius"], errors="coerce")
    df_clean["x0"] = pd.to_numeric(df_clean["x0"], errors="coerce").fillna(0.0)
    df_clean["y0"] = pd.to_numeric(df_clean["y0"], errors="coerce").fillna(0.0)
    df_clean = df_clean.dropna(subset=["Radius"])
    df_clean["Radius"] = df_clean["Radius"].clip(lower=0.0)
    if len(df_clean) > MAX_N:
        st.warning(f"Capped at {MAX_N} circles; extra rows ignored.")
        df_clean = df_clean.head(MAX_N)

    new_radii = df_clean["Radius"].tolist()
    new_x0    = df_clean["x0"].tolist()
    new_y0    = df_clean["y0"].tolist()

    if (new_radii != data["radii"]
            or new_x0   != data["x0"]
            or new_y0   != data["y0"]):
        if len(new_radii) < MIN_N:
            st.warning(f"Need at least {MIN_N} circles to solve.")
        # Validate radii.
        if any(r <= 0 for r in new_radii):
            st.warning("Each radius must be positive (rows with radius ≤ 0 will fail to solve).")
        st.session_state.data = {
            "radii": new_radii,
            "x0": new_x0,
            "y0": new_y0,
        }
        # Invalidate the previous solve since its data may not match.
        st.session_state.res = None
        st.rerun()

    # Action buttons row.
    b1, b2, _ = st.columns([1, 1, 3])
    with b1:
        if st.button("Reset to default (10 unit circles)", width="stretch"):
            st.session_state.data = default_data(DEFAULT_N)
            st.session_state.res = None
            st.session_state.editor_rev += 1  # force editor rebuild
            st.rerun()
    with b2:
        # Uses on_click so the seed bump happens before widgets are
        # instantiated on the next run, which is the only safe time to
        # mutate the seed_input widget's backing key.
        def _randomize_positions():
            radii = st.session_state.data["radii"]
            st.session_state.seed = int(st.session_state.get("seed", 0)) + 1
            if "seed_input" in st.session_state:
                st.session_state["seed_input"] = st.session_state.seed
            xs, ys = random_initial_positions(
                radii, seed=st.session_state.seed
            )
            st.session_state.data["x0"] = xs
            st.session_state.data["y0"] = ys
            st.session_state.editor_rev += 1  # force editor rebuild
        st.button(
            "Randomize initial positions",
            width="stretch",
            on_click=_randomize_positions,
        )


def render_formulation_tab():
    sub_general, sub_instance = st.tabs(["General", "Instance"])

    with sub_general:
        left, right, _ = st.columns([1, 1, 1])
        with left:
            st.markdown(
                "**Sets**  \n"
                r"$\mathcal{I} = \{1, \dots, N\}$ circles"
            )
            st.markdown(
                "**Parameters**  \n"
                r"$r_i$ radius of circle $i \in \mathcal{I}$"
            )
            st.markdown(
                "**Variables**  \n"
                r"$W, H \ge 0$ rectangle width and height" "  \n"
                r"$x_i, y_i \ge 0$ center of circle $i \in \mathcal{I}$"
            )
            st.markdown(
                "**Notes**  \n"
                r"The pairwise non-overlap constraint is *non-convex* "
                r"(its feasible region is the complement of an open ball), "
                r"so the problem typically admits multiple local optima. "
                r"The objective $W \cdot H$ is also non-convex (bilinear). "
                r"The Re-randomize and solve button explores different starting points."
            )
        with right:
            st.markdown(
                r"""<div style="text-align: center;">

**Objective and Constraints**

$$
\begin{aligned}
\min_{W, H, x, y} \quad & W \cdot H \quad \text{(area)} \\
\text{s.t.} \quad & r_i \le x_i \le W - r_i & \forall i \in \mathcal{I} \\
& r_i \le y_i \le H - r_i & \forall i \in \mathcal{I} \\
& (x_i - x_j)^2 + (y_i - y_j)^2 \ge (r_i + r_j)^2 & \forall i < j \\
& W,\, H \ge 0
\end{aligned}
$$

</div>""",
                unsafe_allow_html=True,
            )

    with sub_instance:
        data = st.session_state.data
        n = len(data["radii"])
        radii = data["radii"]

        # Constraint counts.
        n_horiz = 2 * n
        n_vert  = 2 * n
        n_pair  = n * (n - 1) // 2
        n_total = n_horiz + n_vert + n_pair

        st.markdown(
            f"**Instance:** $N = {n}$ circles, "
            f"${2 + 2 * n}$ decision variables "
            f"($W, H$ + $({n} \\times 2)$ centers), "
            f"and ${n_total}$ inequality constraints "
            f"(${n_horiz}$ horizontal fit, ${n_vert}$ vertical fit, "
            f"${n_pair}$ pairwise non-overlap)."
        )

        # Radii table.
        st.markdown("**Radii**")
        radii_df = pd.DataFrame({
            "i": list(range(1, n + 1)),
            "r_i": [f"{r:g}" for r in radii],
        })
        st.dataframe(radii_df, hide_index=True, width="stretch", height=min(300, (n + 1) * 35 + 3))

        # A few representative non-overlap constraints with numeric radii
        # substituted. Shows the first three (i, j) pairs and an ellipsis.
        st.markdown("**Sample non-overlap constraints (numeric radii substituted)**")
        sample_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                sample_pairs.append((i, j))
                if len(sample_pairs) >= 3:
                    break
            if len(sample_pairs) >= 3:
                break

        if sample_pairs:
            rows = []
            for (i, j) in sample_pairs:
                rhs = (radii[i] + radii[j]) ** 2
                rows.append(
                    rf"& (x_{{{i+1}}} - x_{{{j+1}}})^2 + (y_{{{i+1}}} - y_{{{j+1}}})^2 "
                    rf"\ge {rhs:g} \\"
                )
            if n_pair > len(sample_pairs):
                rows.append(r"& \quad \vdots")
            body = r"\begin{aligned}" + "\n".join(rows) + r"\end{aligned}"
            st.latex(body)

        # Objective shown as W * H with current bounds noted.
        st.markdown("**Objective**")
        st.latex(r"\min_{W, H,\, x,\, y} \quad W \cdot H")


def render_logs_tab():
    res = st.session_state.res
    if res is None:
        st.info("Run the optimizer to see solver logs.")
        return
    if res.get("status") == "solver_missing":
        st.error(res.get("message", "rIPOPT not available"))
        return
    log = res.get("log", "") or ""
    if not log.strip():
        st.info("No solver output captured for the last run.")
        return
    st.code(log, language="text")


with tab_opt:
    render_optimizer_tab()
with tab_data:
    render_data_tab()
with tab_form:
    render_formulation_tab()
with tab_logs:
    render_logs_tab()
