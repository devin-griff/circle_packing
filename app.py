# =============================================================================
# Circle Packing — a Streamlit tutorial app.
#
# Pack N non-overlapping circles into the smallest enclosing rectangle. This is
# the textbook non-convex NLP from Biegler's "Nonlinear Programming" — the
# pairwise non-overlap inequalities are non-convex, so multiple local optima
# are expected. The "Randomize ICs" button is the user's tool for exploring
# them.
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
#   - pounce     — the NLP solver (primal-dual interior-point). Called as a
#                  subprocess via Pyomo. Binary ships in the `pyomo-pounce` wheel.
#   - matplotlib — circle plotting (Altair makes circle aspect ratios painful).
#
# File roadmap (mirrors strip-packing):
#   1. Imports + module-level constants (defaults, palette).
#   2. solve()                  — builds and solves the Pyomo NLP.
#   3. State helpers            — init_state, apply_reset, add/remove,
#                                 randomize_ics.
#   4. Layout helpers           — _render_top_metric, build_packing_fig.
#   5. Tab renderers            — Optimizer / Formulation / Logs.
#   6. Main                     — page config, header, tab assembly.
# =============================================================================

import base64
import contextlib
import copy
import io
import math
import time
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pyomo.environ as pyo
# Registers `pounce` with pyo.SolverFactory via decorator side-effect; the
# wheel also bundles the solver binary so no system install is required.
import pyomo_pounce  # noqa: F401
from pyomo.common.errors import ApplicationError


# ---------- Constants ----------

# Circle count limits. Pounce / IPOPT-family NLPs scale well, but the editor
# itself gets cramped beyond ~20 rows on a single column.
MIN_N = 1
MAX_N = 20

# Default instance: ten circles with a mix of radii (1, 2, 3) laid out on a
# loose grid. The grid cell is sized to the LARGEST radius so even the big
# circles don't overlap their neighbors at the initial-condition view, and
# `gap > 0` between cell edges leaves the optimizer obvious room to shrink.
# Integer spacing keeps the +/- step-of-1 stepper UI happy.
def _default_grid_positions(radii, gap=2):
    """Lay out len(radii) circles on a square grid. Each cell is sized to
    fit the largest radius with `gap` units between cell edges, so any
    circle (regardless of its own radius) sits comfortably inside its
    cell. Returns a list of (cx, cy) tuples aligned to `radii`'s order."""
    n = len(radii)
    if n == 0:
        return []
    cols = int(math.ceil(math.sqrt(n)))
    rmax = max(radii)
    cell = 2 * rmax + gap
    return [
        ((i % cols) * cell + rmax,
         (i // cols) * cell + rmax)
        for i in range(n)
    ]

_DEFAULT_RADII = [2, 1, 1, 3, 1, 2, 1, 1, 2, 1]
_DEFAULT_N = len(_DEFAULT_RADII)
_default_positions = _default_grid_positions(_DEFAULT_RADII)
# Stored as floats (with .0 tails for integer defaults) so the number_input
# steppers stay in float mode — the steppers move by 1.0 but values can be
# fractional after a solve syncs the optimal x/y back into x0/y0.
DEFAULT_DATA = {
    "circles": list(range(1, _DEFAULT_N + 1)),
    "r":  {i: float(_DEFAULT_RADII[i - 1]) for i in range(1, _DEFAULT_N + 1)},
    "x0": {i: float(_default_positions[i - 1][0]) for i in range(1, _DEFAULT_N + 1)},
    "y0": {i: float(_default_positions[i - 1][1]) for i in range(1, _DEFAULT_N + 1)},
}

# A 12-color categorical palette repeated as needed. Tableau-style; matches
# strip-packing's editor index badges for cross-app visual consistency. The
# plot itself uses matplotlib's tab10/tab20 colormap so adjacent indices
# stay distinguishable at small circle sizes.
_PALETTE = [
    "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#EECA3B",
    "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC", "#1F77B4", "#9467BD",
]


# ---------- Solver ----------
#
# `solve` builds and solves the circle-packing NLP. The model is plain Pyomo:
# sets, vars, bounds, the bilinear W*H objective, and the pairwise non-overlap
# constraints. We capture stdout so pounce's output lands in the Logs tab.

def solve(data):
    """Build and solve the circle-packing NLP for the given data dict.
    Returns a result dict with W, H, x{cid}, y{cid}, area, status, log, elapsed."""
    circles = data["circles"]
    n = len(circles)
    if n < 1:
        return {
            "status": "no_circles", "log": "", "W": None, "H": None,
            "x": {}, "y": {}, "area": None, "elapsed": 0.0,
        }

    m = pyo.ConcreteModel()
    m.I = pyo.Set(initialize=circles, ordered=True)
    m.r = pyo.Param(m.I, initialize={i: float(data["r"][i]) for i in circles})

    # Upper bound for the rectangle and the centers. 2 * sum(r) is generous
    # (any reasonable packing fits in that square), and keeps the interior-
    # point solver well-conditioned.
    sum_r = sum(float(data["r"][i]) for i in circles)
    max_r = max(float(data["r"][i]) for i in circles)
    big = max(2.0 * sum_r, 4.0 * max_r)

    m.W = pyo.Var(bounds=(0.0, big), initialize=2.0 * sum_r)
    m.H = pyo.Var(bounds=(0.0, big), initialize=2.0 * sum_r)
    m.x = pyo.Var(m.I, bounds=(0.0, big))
    m.y = pyo.Var(m.I, bounds=(0.0, big))

    # User-supplied initial guess for the centers. The objective and W/H
    # init to the generous box, then the solver pushes them down.
    for i in circles:
        m.x[i].value = float(data["x0"][i])
        m.y[i].value = float(data["y0"][i])

    # Inside-rectangle: each circle fits horizontally and vertically.
    m.x_upper = pyo.Constraint(m.I, rule=lambda m, i: m.x[i] + m.r[i] <= m.W)
    m.x_lower = pyo.Constraint(m.I, rule=lambda m, i: m.x[i] >= m.r[i])
    m.y_upper = pyo.Constraint(m.I, rule=lambda m, i: m.y[i] + m.r[i] <= m.H)
    m.y_lower = pyo.Constraint(m.I, rule=lambda m, i: m.y[i] >= m.r[i])

    # Pairwise non-overlap. Squared form so the constraint stays smooth at
    # the boundary (sqrt would have a non-differentiable point at distance
    # zero). This is the non-convex piece of the model.
    def no_overlap(m, i, j):
        if i >= j:
            return pyo.Constraint.Skip
        return (m.x[i] - m.x[j])**2 + (m.y[i] - m.y[j])**2 \
               >= (m.r[i] + m.r[j])**2
    m.no_overlap = pyo.Constraint(m.I, m.I, rule=no_overlap)

    # Objective: minimize area. Bilinear W*H — also non-convex, but the
    # interior-point solver handles it fine.
    m.area = pyo.Objective(expr=m.W * m.H, sense=pyo.minimize)

    buf = io.StringIO()
    solver = pyo.SolverFactory("pounce")
    t0 = time.perf_counter()
    try:
        with contextlib.redirect_stdout(buf):
            results = solver.solve(m, tee=True)
    except ApplicationError as e:
        return {
            "status": "solver_missing",
            "message": (
                "pounce solver binary not found. The `pyomo-pounce` wheel "
                "should bundle it — check your environment. "
                f"({e})"
            ),
            "log": buf.getvalue(),
            "W": None, "H": None, "x": {}, "y": {}, "area": None,
            "elapsed": time.perf_counter() - t0,
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Solver error: {e}",
            "log": buf.getvalue(),
            "W": None, "H": None, "x": {}, "y": {}, "area": None,
            "elapsed": time.perf_counter() - t0,
        }
    elapsed = time.perf_counter() - t0

    tc = results.solver.termination_condition
    status = str(tc)

    # Pull values whether or not the status is "optimal" — pounce often
    # returns a usable layout on "locallyOptimal" or "maxIterations".
    try:
        W_val = float(pyo.value(m.W))
        H_val = float(pyo.value(m.H))
        xs = {i: float(pyo.value(m.x[i])) for i in circles}
        ys = {i: float(pyo.value(m.y[i])) for i in circles}
        area_val = float(pyo.value(m.area))
    except Exception:
        return {
            "status": status, "log": buf.getvalue(),
            "W": None, "H": None, "x": {}, "y": {}, "area": None,
            "elapsed": elapsed,
        }

    return {
        "status": status, "log": buf.getvalue(),
        "W": W_val, "H": H_val, "x": xs, "y": ys, "area": area_val,
        "elapsed": elapsed,
        # Snapshot the data that produced this result so the renderer can
        # decide whether the optimal layout still matches what's in the
        # editor (plot reverts to initial-condition view if the user has
        # edited a value, added/removed a circle, or randomized ICs since
        # this solve). The metrics row keeps showing these values until
        # the next Solve / Reset, so the user has a stable readout.
        "data_at_solve": copy.deepcopy(data),
    }


# ---------- State helpers ----------

def init_state():
    # Idempotent — only seed defaults the first time, otherwise the user's
    # edits get wiped on every rerun.
    if "data" not in st.session_state:
        st.session_state.data = copy.deepcopy(DEFAULT_DATA)
    if "optimal" not in st.session_state:
        st.session_state.optimal = None
    if "seed" not in st.session_state:
        st.session_state.seed = 0
    # The reset button can't directly mutate widget-backed keys without
    # raising a Streamlit error, so it sets a flag and reruns. We apply
    # the reset here, before widgets are instantiated this run.
    if st.session_state.pop("_pending_reset", False):
        apply_reset()


def apply_reset():
    st.session_state.data = copy.deepcopy(DEFAULT_DATA)
    st.session_state.optimal = None
    st.session_state.seed = 0
    # Bump the circle-editor widget version so all per-circle steppers
    # re-init from data instead of holding onto sticky pre-reset values.
    st.session_state["_circle_editor_ver"] = (
        st.session_state.get("_circle_editor_ver", 0) + 1
    )


# Circles are tracked by stable opaque integer ids — `circles` is a list of
# ids; `r`, `x0`, `y0` map id → value. Ids don't renumber on delete so the
# editor widgets don't get reassigned to a different circle when one is
# removed.

def add_circle(data, r=1.0, x0=0.0, y0=0.0):
    new_id = (max(data["circles"]) + 1) if data["circles"] else 1
    data["circles"] = list(data["circles"]) + [new_id]
    data["r"]  = dict(data["r"]);  data["r"][new_id]  = float(r)
    data["x0"] = dict(data["x0"]); data["x0"][new_id] = float(x0)
    data["y0"] = dict(data["y0"]); data["y0"][new_id] = float(y0)
    return data


def remove_circle(data, cid):
    data["circles"] = [i for i in data["circles"] if i != cid]
    data["r"]  = {i: v for i, v in data["r"].items()  if i != cid}
    data["x0"] = {i: v for i, v in data["x0"].items() if i != cid}
    data["y0"] = {i: v for i, v in data["y0"].items() if i != cid}
    return data


def _snap_step(old_val, new_val, step=1.0):
    """If `old_val` is non-integer and `new_val` looks like exactly one
    +step or -step click away from it (i.e. the user tapped the editor's
    +/- button), snap to the nearest integer in that direction —
    `ceil(old)` for +, `floor(old)` for -. Lets the user start from a
    fractional optimal x/y (synced into the editor after a solve) and
    reach a round integer in a single click instead of dragging along a
    fractional offset. Old values that are already integer pass through
    untouched, as do typed-in edits (which don't look like a step away)."""
    if abs(old_val - round(old_val)) < 1e-9:
        return new_val
    diff = new_val - old_val
    if abs(diff - step) < step / 2:
        return float(math.ceil(old_val))
    if abs(diff + step) < step / 2:
        return float(math.floor(old_val))
    return new_val


def _snap_pre_render(widget_key, data_val, step=1.0):
    """Apply the snap *before* the widget is rendered. Streamlit's
    number_input reads its current value from `st.session_state[key]` if
    present (the `value=` arg is only the initial value); writing the
    snapped value into session_state here means the widget renders
    directly at the snapped value, not the raw +/- step. Without this,
    a + click on 12.3 visibly flashes 13.3 for one frame before the
    post-render snap kicks in and reruns with 13.0."""
    if widget_key not in st.session_state:
        return
    snapped = _snap_step(data_val, st.session_state[widget_key], step)
    if snapped != st.session_state[widget_key]:
        st.session_state[widget_key] = snapped


def randomize_ics(data, seed):
    """In-place: re-roll (x0, y0) for all circles using a deterministic seed.
    Radii are left alone. Centers are sampled at integer coordinates inside
    a square of side max(2*sum(r), 4*max(r)) — large enough to start far
    from optimal in most configurations, small enough that the solver
    doesn't wander. Stored as floats so the editor's float-mode steppers
    are happy (the steppers move by 1.0 but values can be fractional after
    a solve syncs the optimal positions back into x0/y0)."""
    rng = np.random.default_rng(seed)
    circles = data["circles"]
    if not circles:
        return data
    radii = [float(data["r"][i]) for i in circles]
    side = int(round(max(2.0 * sum(radii), 4.0 * max(radii))))
    xs = rng.integers(0, side + 1, size=len(circles))
    ys = rng.integers(0, side + 1, size=len(circles))
    data["x0"] = {cid: float(xs[k]) for k, cid in enumerate(circles)}
    data["y0"] = {cid: float(ys[k]) for k, cid in enumerate(circles)}
    return data


# ---------- Status helper ----------

def status_label(status):
    """Map raw Pyomo termination conditions onto a short user-facing label
    plus a streamlit-style severity ("ok" | "warn" | "err")."""
    s = (status or "").lower().replace(" ", "")
    if s == "optimal":
        return "Optimal", "ok"
    if "locallyoptimal" in s:
        return "Local opt.", "ok"
    if "feasible" in s and "infeasible" not in s:
        return "Feasible", "warn"
    if "maxiterations" in s or "iteration" in s:
        return "Max iter", "warn"
    if "infeasible" in s:
        return "Infeasible", "err"
    if "unbounded" in s:
        return "Unbounded", "err"
    if s in ("solver_missing", "error", "no_circles"):
        return status, "err"
    return status, "warn"


# ---------- Layout helpers ----------

def _render_top_metric(slot, label, value, suffix_html=""):
    """Render a metric-shaped block via raw HTML. Mirrors the strip-packing
    helper (which in turn mirrors diet's colored_metric): small gray label
    on top, large value below, with an optional HTML suffix appended inside
    the value div (used to drop a red ⚠ next to "Status" when pounce
    returned a non-OK termination). `white-space: nowrap` on both label and
    value keeps each metric on a single line."""
    slot.markdown(
        "<div style='margin:0.25rem 0 1rem 0; line-height:1.2;'>"
        "<div style='font-size:0.875rem; color:rgba(49,51,63,0.6); "
        "margin-bottom:0.25rem; white-space:nowrap;'>"
        f"{label}"
        "</div>"
        "<div style='font-size:2.25rem; font-weight:400; line-height:1.2; "
        "white-space:nowrap;'>"
        f"{value}{suffix_html}"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def _bounding_box(circles, r, x, y):
    """Smallest axis-aligned bounding box around the given circles.
    Returns (xmin, ymin, xmax, ymax)."""
    if not circles:
        return (0.0, 0.0, 1.0, 1.0)
    xmin = min(x[i] - r[i] for i in circles)
    xmax = max(x[i] + r[i] for i in circles)
    ymin = min(y[i] - r[i] for i in circles)
    ymax = max(y[i] + r[i] for i in circles)
    return (xmin, ymin, xmax, ymax)


def build_packing_fig(data, layout, *, mode):
    """Draw the rectangle + circles to a matplotlib Figure.

    `layout` is one of:
      - mode="optimal": dict with W, H, x{cid}, y{cid} — red dashed rectangle
        at (0,0,W,H) is drawn around the optimized circle positions.
      - mode="initial": dict with x{cid}, y{cid} — a *gray* dashed rectangle
        is drawn around the smallest bounding box that fits the initial
        circles, so the user has a clear "starting state" visual."""
    fig, ax = plt.subplots(figsize=(7, 7))
    circles = data["circles"]
    rs = data["r"]
    xs = layout["x"]
    ys = layout["y"]

    if mode == "optimal":
        W = float(layout["W"])
        H = float(layout["H"])
        rect_x, rect_y = 0.0, 0.0
        rect_w, rect_h = W, H
        rect_color = "#dc2626"   # red — same accent as strip-packing's strip outline
    else:
        xmin, ymin, xmax, ymax = _bounding_box(circles, rs, xs, ys)
        rect_x, rect_y = xmin, ymin
        rect_w, rect_h = (xmax - xmin) or 1.0, (ymax - ymin) or 1.0
        rect_color = "#9ca3af"   # gray — "this is your starting state, not the answer"

    ax.add_patch(mpatches.Rectangle(
        (rect_x, rect_y), rect_w, rect_h,
        linewidth=2, edgecolor=rect_color, facecolor="none", linestyle="--",
    ))

    n = len(circles)
    cmap = plt.get_cmap("tab20" if n > 10 else "tab10")
    for display_idx, cid in enumerate(circles, start=1):
        color = cmap((display_idx - 1) % cmap.N)
        ax.add_patch(mpatches.Circle(
            (xs[cid], ys[cid]), rs[cid],
            linewidth=1.5, edgecolor="black", facecolor=color, alpha=0.55,
        ))
        ax.text(
            xs[cid], ys[cid], str(display_idx),
            ha="center", va="center", fontsize=9, fontweight="bold",
            color="black",
        )

    pad = 0.08 * max(rect_w, rect_h, 1.0)
    ax.set_xlim(rect_x - pad, rect_x + rect_w + pad)
    ax.set_ylim(rect_y - pad, rect_y + rect_h + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig


# ---------- Tab renderers ----------

def _render_circle_editor(data):
    """The circle editor — one row per circle with three stepper inputs
    (radius, x₀, y₀) and a delete button. Used in the left column of the
    Optimizer tab. Mirrors strip-packing's _render_rect_editor."""
    st.markdown(f"#### Circles (max {MAX_N})")

    ver = st.session_state.get("_circle_editor_ver", 0)
    editor_cols = [0.5, 1.2, 1.2, 1.2, 0.6]

    header = st.columns(editor_cols)
    header[0].markdown("")
    header[1].markdown("**Radius**")
    header[2].markdown("**x₀**")
    header[3].markdown("**y₀**")
    header[4].markdown("")

    new_data = None
    for display_idx, cid in enumerate(data["circles"], start=1):
        cols = st.columns(editor_cols, vertical_alignment="center")
        color = _PALETTE[(display_idx - 1) % len(_PALETTE)]
        cols[0].markdown(
            f'<div style="display:inline-flex;align-items:center;'
            f'justify-content:center;width:1.6rem;height:1.6rem;'
            f'border-radius:0.3rem;background:{color};color:#fff;'
            f'font-weight:700;font-size:0.85rem;">{display_idx}</div>',
            unsafe_allow_html=True,
        )
        # Pre-render snap: if the widget's stored value looks like a +/-
        # click off a fractional data value, rewrite session_state to the
        # snapped value before the widget renders. The widget then shows
        # the snapped value directly (no visible 13.3 flash before
        # settling on 13). See _snap_pre_render.
        _snap_pre_render(f"r_{cid}_{ver}",  float(data["r"][cid]))
        _snap_pre_render(f"x0_{cid}_{ver}", float(data["x0"][cid]))
        _snap_pre_render(f"y0_{cid}_{ver}", float(data["y0"][cid]))
        # Float mode with step=1.0 so the +/- buttons increment by 1, but
        # values can carry a fractional part — after a solve we write the
        # optimal x/y back into the editor (rounded to 1 decimal), and the
        # user can then perturb by clean integer offsets.
        new_r = cols[1].number_input(
            "Radius", min_value=1.0, max_value=20.0, step=1.0, format="%.1f",
            value=float(data["r"][cid]),
            key=f"r_{cid}_{ver}", label_visibility="collapsed",
        )
        new_x = cols[2].number_input(
            "x0", min_value=-100.0, max_value=100.0, step=1.0, format="%.1f",
            value=float(data["x0"][cid]),
            key=f"x0_{cid}_{ver}", label_visibility="collapsed",
        )
        new_y = cols[3].number_input(
            "y0", min_value=-100.0, max_value=100.0, step=1.0, format="%.1f",
            value=float(data["y0"][cid]),
            key=f"y0_{cid}_{ver}", label_visibility="collapsed",
        )
        if cols[4].button("🗑", key=f"del_{cid}_{ver}"):
            st.session_state.data = remove_circle(dict(data), cid)
            st.rerun()
        if (new_r != data["r"][cid]
                or new_x != data["x0"][cid]
                or new_y != data["y0"][cid]):
            new_data = dict(data)
            new_data["r"]  = dict(new_data["r"]);  new_data["r"][cid]  = new_r
            new_data["x0"] = dict(new_data["x0"]); new_data["x0"][cid] = new_x
            new_data["y0"] = dict(new_data["y0"]); new_data["y0"][cid] = new_y

    if new_data is not None:
        st.session_state.data = new_data
        st.rerun()

    can_add = len(data["circles"]) < MAX_N
    btn_cols = st.columns(editor_cols)
    with btn_cols[1]:
        if st.button(
            "➕ Add circle",
            key="circles_add",
            disabled=not can_add,
            help=(
                None if can_add
                else f"Max {MAX_N} circles (the editor gets cramped beyond this)."
            ),
        ):
            st.session_state.data = add_circle(dict(data))
            st.rerun()
    with btn_cols[2]:
        if st.button(
            "Reset to defaults",
            key="circles_reset",
            help=f"Restore the default {_DEFAULT_N}-circle instance.",
        ):
            st.session_state["_pending_reset"] = True
            st.rerun()


def _fill_metric_slots(w_slot, h_slot, area_slot, status_slot, data, optimal):
    """Render the four top-row metrics from the current (data, optimal)
    state. Called BOTH before and after the solve handler so the row height
    stays locked in during the spin — otherwise empty metric placeholders
    collapse to 0 height, the row shrinks, and the bottom-aligned
    Solve/Randomize buttons jump up while pounce is running, then drop back
    down when the metrics refill. Strip-packing dodges this naturally
    because its W stepper / transform radio always-take-height anchor the
    row, but circle-packing's top row is all-buttons-plus-metric-slots."""
    if not data["circles"]:
        _render_top_metric(w_slot, "Width", "—")
        _render_top_metric(h_slot, "Height", "—")
        _render_top_metric(area_slot, "Area", "—")
        _render_top_metric(status_slot, "Status", "—")
        return

    has_optimal = bool(
        optimal
        and optimal["status"] not in ("solver_missing", "error", "no_circles")
        and optimal.get("W") is not None
    )
    # When the current data matches what was solved, show the optimal
    # W/H/Area + the solver status. Otherwise (no solve yet, OR data has
    # changed since the last solve) show the initial-condition bounding
    # box — same rectangle the gray dashed outline traces in the plot —
    # with Status "Initialized" so the row always describes whatever the
    # plot is currently showing.
    fresh_optimal = has_optimal and optimal.get("data_at_solve") == data

    if fresh_optimal:
        _render_top_metric(w_slot, "Width", f"{optimal['W']:.1f}")
        _render_top_metric(h_slot, "Height", f"{optimal['H']:.1f}")
        _render_top_metric(area_slot, "Area", f"{optimal['area']:.1f}")
        sev_label, severity = status_label(optimal["status"])
        if severity == "ok":
            _render_top_metric(status_slot, "Status", sev_label)
        else:
            tooltip = (
                f"pounce termination: {optimal['status']}. "
                "Pounce is a local solver and the problem is non-convex, "
                "so this run didn't find a clean local minimum. "
                "Try Randomize ICs."
            )
            violation_icon = (
                '<span class="circle-violation-icon" '
                f'data-violation-tooltip="{tooltip}" '
                'style="color:#dc2626; cursor:default; font-weight:700; '
                'margin-left:0.4em; vertical-align:baseline;">⚠</span>'
            )
            _render_top_metric(
                status_slot, "Status", sev_label,
                suffix_html=violation_icon,
            )
    else:
        xmin, ymin, xmax, ymax = _bounding_box(
            data["circles"], data["r"], data["x0"], data["y0"]
        )
        w = xmax - xmin
        h = ymax - ymin
        _render_top_metric(w_slot, "Width", f"{w:.1f}")
        _render_top_metric(h_slot, "Height", f"{h:.1f}")
        _render_top_metric(area_slot, "Area", f"{w * h:.1f}")
        _render_top_metric(status_slot, "Status", "Initialized")


def render_optimizer_tab():
    # Page-wide CSS for editor steppers (tight spacing) + the red ⚠
    # tooltip pattern reused from strip-packing.
    st.markdown(
        """
        <style>
        [data-testid="stMainBlockContainer"]
            [data-testid="stHorizontalBlock"] {
            margin-bottom: -0.75rem;
        }
        [data-testid="stNumberInputContainer"] input {
            padding-top: 0.25rem; padding-bottom: 0.25rem;
            text-align: right; padding-right: 0.4rem;
        }
        /* Hide the fullscreen-toggle button that Streamlit overlays on
           charts on hover. Match by title only — using broader
           data-testid selectors like stElementToolbar accidentally hid
           the wrapper the matplotlib chart renders inside, which left
           the chart as a broken <img> placeholder on production. */
        button[title="View fullscreen"],
        button[title*="ullscreen" i] {
            display: none !important;
        }
        .circle-violation-icon {
            position: relative;
            display: inline-block;
        }
        .circle-violation-icon:hover::after {
            content: attr(data-violation-tooltip);
            position: absolute;
            top: 100%;
            left: 0;
            margin-top: 0.25rem;
            background: #000;
            color: #fff;
            padding: 0.5rem 0.75rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-family: inherit;
            font-weight: 400;
            line-height: 1.4;
            width: max-content;
            max-width: 24rem;
            white-space: normal;
            z-index: 1000;
            pointer-events: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    data = st.session_state.data
    optimal = st.session_state.optimal

    # Two-column layout: editor (left) | plot column fills remainder. Editor
    # gets a touch more room than strip-packing (4 vs 3) because we have
    # three numeric inputs per row instead of two.
    editor_col, plot_col = st.columns([4, 8])

    with editor_col:
        _render_circle_editor(data)

    with plot_col:
        # Controls + metrics in one row above the plot. Solve sits far left,
        # then Randomize, then 4 metric slots. Status gets extra weight so
        # the "Local opt." text fits comfortably. Plot renders below through
        # a placeholder so the controls in the top row can update
        # session_state before we paint.
        top_row = st.columns(
            [1, 1.4, 1, 1, 1, 1.4],
            vertical_alignment="bottom",
        )
        with top_row[0]:
            solve_clicked = st.button(
                "Solve", type="primary", use_container_width=True,
            )
        with top_row[1]:
            randomize_clicked = st.button(
                "Randomize ICs", use_container_width=True,
                help="Re-roll initial guesses (x₀, y₀) for all circles. "
                     "Radii stay put.",
            )
        w_slot = top_row[2].empty()
        h_slot = top_row[3].empty()
        area_slot = top_row[4].empty()
        status_slot = top_row[5].empty()

        # Fill metric slots BEFORE the solve handler so the row keeps its
        # full height during the spin. See _fill_metric_slots docstring.
        _fill_metric_slots(w_slot, h_slot, area_slot, status_slot, data, optimal)

        plot_slot = st.empty()
        # Dedicated spinner slot so the spinner appears just below the plot
        # without replacing it. st.empty() collapses to zero height when
        # nothing's in it, so layout is unaffected at rest.
        spinner_slot = st.empty()

    # ── Handle button clicks ────────────────────────────────────────────────
    if randomize_clicked:
        st.session_state.seed = int(st.session_state.get("seed", 0)) + 1
        randomize_ics(st.session_state.data, st.session_state.seed)
        st.session_state["_circle_editor_ver"] = (
            st.session_state.get("_circle_editor_ver", 0) + 1
        )
        st.rerun()

    if solve_clicked:
        with spinner_slot.container():
            with st.spinner("Running pounce optimization..."):
                result = solve(data)
        spinner_slot.empty()
        # Sync the editor's x0/y0 to the optimal positions (rounded to 1
        # decimal) so the user can perturb from there with the integer
        # stepper. We must also re-stamp result["data_at_solve"] to the
        # synced data so the Status row still reads "Local opt." — the
        # solved-vs-current data comparison includes these positions.
        if result.get("W") is not None and result.get("x"):
            new_data = dict(st.session_state.data)
            new_data["x0"] = dict(new_data["x0"])
            new_data["y0"] = dict(new_data["y0"])
            for cid in new_data["circles"]:
                if cid in result["x"]:
                    new_data["x0"][cid] = round(float(result["x"][cid]), 1)
                    new_data["y0"][cid] = round(float(result["y"][cid]), 1)
            st.session_state.data = new_data
            result = dict(result)
            result["data_at_solve"] = copy.deepcopy(new_data)
            st.session_state["_circle_editor_ver"] = (
                st.session_state.get("_circle_editor_ver", 0) + 1
            )
        st.session_state.optimal = result
        # Rerun so the editor (rendered earlier this run with the pre-solve
        # ver / pre-solve x0,y0) gets repainted under the post-solve ver
        # with the synced optimal values. Without this, the first +/- click
        # after a solve hits the stale widget and is effectively eaten,
        # which is why the user was needing two clicks.
        st.rerun()

    # ── Paint plot ──────────────────────────────────────────────────────────
    if not data["circles"]:
        with plot_slot.container():
            st.info("Add at least one circle on the left to set up the problem.")
        return

    has_optimal = bool(
        optimal
        and optimal["status"] not in ("solver_missing", "error", "no_circles")
        and optimal.get("W") is not None
    )
    # Optimal LAYOUT shows only when the current data matches what was
    # solved. After any edit / randomize / add / delete, fall back to the
    # initial-condition view. (The metrics row still shows the last solve's
    # values — see _fill_metric_slots calls above.)
    plot_optimal = has_optimal and optimal.get("data_at_solve") == data

    with plot_slot.container():
        if plot_optimal:
            fig = build_packing_fig(
                data,
                {"W": optimal["W"], "H": optimal["H"],
                 "x": optimal["x"], "y": optimal["y"]},
                mode="optimal",
            )
        else:
            fig = build_packing_fig(
                data,
                {"x": data["x0"], "y": data["y0"]},
                mode="initial",
            )
        # Figure fills the plot column at use_container_width so the
        # spinner_slot below it lines up at the same width — same pattern
        # strip-packing uses for its strip.
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Status caption beneath the plot — only on non-OK outcomes, and
        # only while the plotted layout actually IS the latest solve.
        # Once the user edits data, the plot reverts to initial-condition
        # mode and an "infeasible" warning about a stale solve would be
        # misleading.
        if plot_optimal:
            sev_label, severity = status_label(optimal["status"])
            if severity == "err":
                st.error(optimal.get("message") or f"Solver status: {sev_label}")
            elif severity == "warn":
                st.warning(
                    f"Solver status: {sev_label} — results may be suboptimal. "
                    "Try **Randomize ICs** to explore other local minima."
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

        st.markdown("---")
        # Heading and paragraph in a single markdown so Streamlit doesn't
        # insert its default block margin between the two — keeps the
        # Solution method title visually attached to its prose.
        st.markdown(
            "**Solution method**  \n"
            "Solved as a non-convex NLP with **pounce**, a primal-dual "
            "interior-point solver from John Kitchin, distributed as a "
            "pip wheel via `pyomo-pounce`. Interior-point methods follow "
            "a barrier path through the interior of the feasible region "
            "and converge to a *local* minimum near the initial guess. "
            "The pairwise non-overlap inequalities and the bilinear "
            r"$W \cdot H$ objective are both non-convex, so different "
            "starting points often land at different local optima. The "
            r"Optimizer tab's **Randomize ICs** button re-rolls $(x_0, "
            r"y_0)$ for every circle to explore alternate basins; after "
            "each solve the editor writes the locally-optimal positions "
            r"back into the $(x_0, y_0)$ fields so you can perturb a "
            "known optimum by hand and re-solve to test how sensitive it is."
        )
        st.markdown(
            "See the [companion Jupyter notebook]"
            "(https://github.com/devin-griff/circle_packing/blob/main/Circle%20packing.ipynb) "
            "for the Pyomo implementation."
        )

        st.markdown("**References**")
        st.markdown(
            "[1] L. T. Biegler, *Nonlinear Programming: Concepts, Algorithms, "
            "and Applications to Chemical Processes*. Philadelphia, PA: SIAM, "
            "2010. "
            "[SIAM](https://epubs.siam.org/doi/book/10.1137/1.9780898719383)"
        )
        st.markdown(
            "[2] M. L. Bynum, G. A. Hackebeil, W. E. Hart, C. D. Laird, "
            "B. L. Nicholson, J. D. Siirola, J.-P. Watson, and D. L. Woodruff, "
            "*Pyomo — Optimization Modeling in Python*, 3rd ed. "
            "Cham: Springer, 2021. "
            "[Springer](https://link.springer.com/book/10.1007/978-3-030-68928-5)"
        )

    with sub_instance:
        st.subheader("Instance Summary")
        data = st.session_state.data
        if not data["circles"]:
            st.info("Add at least one circle on the Optimizer tab.")
            return

        circles = data["circles"]
        N = len(circles)
        radii = [float(data["r"][i]) for i in circles]
        sum_r = sum(radii)
        sum_r2 = sum(r * r for r in radii)
        rmax = max(radii)
        total_circle_area = math.pi * sum_r2
        # Geometric lower bound on rectangle area: must contain every
        # circle (so both W and H are at least 2*rmax, giving area >=
        # 4*rmax^2) AND must enclose every circle's interior (so area
        # >= sum of circle areas = pi * sum(r^2)).
        area_lb = max(4.0 * rmax * rmax, total_circle_area)
        n_pair = N * (N - 1) // 2

        st.markdown(
            f"**N (circles)** &nbsp; {N}  \n"
            f"**Sum of radii $\\sum_i r_i$** &nbsp; {sum_r:g}  \n"
            f"**Total circle area $\\pi \\sum_i r_i^2$** &nbsp; "
            f"{total_circle_area:.1f}  \n"
            f"**Lower bound on $W \\cdot H$** &nbsp; "
            f"$\\max(4 r_{{\\max}}^2,\\ \\pi \\sum_i r_i^2) = "
            f"{area_lb:.1f}$  \n"
            f"**Non-overlap constraints $N(N-1)/2$** &nbsp; {n_pair}"
        )

        if N >= 2:
            # Worked non-overlap constraint for the first pair with this
            # instance's actual radii substituted in. The General sub-tab
            # carries the abstract form; here we ground it in the user's
            # data so the (r_i + r_j)^2 right-hand side is a concrete
            # number rather than a symbol.
            i_, j_ = 1, 2
            ri, rj = radii[0], radii[1]
            rhs = (ri + rj) ** 2
            st.markdown("---")
            st.markdown(
                rf"**Non-overlap constraint (instantiated)** &nbsp; for the "
                rf"pair $({i_},\,{j_})$ with $r_{i_}={ri:g}$ and "
                rf"$r_{j_}={rj:g}$, the squared center-to-center distance "
                "must be at least the squared sum of radii:"
            )
            st.latex(
                rf"(x_{i_} - x_{j_})^2 + (y_{i_} - y_{j_})^2 \ge "
                rf"({ri:g} + {rj:g})^2 = {rhs:g}"
            )
            st.caption(
                f"This is one of the {n_pair} pairwise non-overlap "
                "constraints in the model. The squared form keeps the "
                "constraint smooth at distance zero (a $\\sqrt{\\cdot}$ "
                "would have a non-differentiable point there), which the "
                "interior-point solver needs."
            )


def render_logs_tab():
    optimal = st.session_state.optimal
    if optimal is None:
        st.info("Run the optimizer to see solver logs.")
        return
    if optimal.get("status") == "solver_missing":
        st.error(optimal.get("message", "pounce not available"))
        return
    log = optimal.get("log", "") or ""
    if not log.strip():
        st.info("No solver output captured for the last run.")
        return
    st.code(log, language="text")


# ---------- Main ----------
#
# Module-level code runs on every Streamlit rerun, so this section needs to be
# cheap and idempotent: configure the page, ensure session_state is set up,
# inject the fixed-corner home-logo CSS, render the header/caption, then
# assemble the three tabs.

st.set_page_config(
    page_title="Circle Packing",
    page_icon="favicon.png",
    layout="wide",
)

init_state()

# Fixed-corner home logo (no sidebar in this app — all controls inline on
# the Optimizer tab). Same pattern as strip-packing / diet / knapsack.
_FAVICON_DATA_URL = "data:image/png;base64," + base64.b64encode(
    (Path(__file__).parent / "favicon.png").read_bytes()
).decode()
st.markdown(
    """
    <style>
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
    /* Top padding shared across the template family — clears the sticky
       header without clipping the title. */
    .block-container,
    [data-testid="stMainBlockContainer"] {
        padding-top: 2.5rem !important;
        padding-bottom: 0rem !important;
    }
    /* Streamlit pins a tiny « collapse-sidebar arrow at the top of the
       (now-absent) sidebar area; hide it since we don't have a sidebar. */
    [data-testid="stSidebarHeader"] {
        display: none;
    }
    </style>
    """
    f'<a href="https://griffith-pse.com" target="_self" '
    f'class="home-logo-corner">'
    f'<img src="{_FAVICON_DATA_URL}" alt="Griffith PSE — home" />'
    f"</a>",
    unsafe_allow_html=True,
)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    "<h2 style='margin: 0 0 0.25rem 0; padding: 0; font-size: 1.5rem; font-weight: 700;'>"
    "Circle Packing "
    "<span style='font-size: 1.15rem; font-weight: 400; color: #6b7280;'>"
    "powered by "
    "<a href='https://github.com/Pyomo/pyomo' target='_blank' "
    "style='color: #6b7280; text-decoration: underline;'>Pyomo</a>"
    " + "
    "<a href='https://github.com/jkitchin/pounce' target='_blank' "
    "style='color: #6b7280; text-decoration: underline;'>Pounce</a>"
    "</span>"
    "</h2>",
    unsafe_allow_html=True,
)
_caption_col, _ = st.columns([5, 3])
with _caption_col:
    st.markdown(
        "Pack $N$ non-overlapping circles into the smallest enclosing "
        "rectangle. This is a classic non-convex NLP from Prof. Biegler's "
        "*Nonlinear Programming* textbook — the pairwise non-overlap "
        "inequalities are non-convex, so different starting points often "
        "land at different local optima. Edit the circle list (radius + "
        "initial guess $(x_0, y_0)$) on the Optimizer tab and click **Solve**; "
        "use **Randomize ICs** to re-roll initial guesses and explore "
        "alternate local minima. The **📐 Formulation** tab walks through "
        "the underlying NLP, the interior-point solution method, and the "
        "references; the **📜 Logs** tab shows the raw pounce output from "
        "the latest solve."
    )

# Three tabs.
optimizer_tab, formulation_tab, logs_tab = st.tabs(
    ["🎯 Optimizer", "📐 Formulation", "📜 Logs"]
)

with optimizer_tab:
    render_optimizer_tab()
with formulation_tab:
    render_formulation_tab()
with logs_tab:
    render_logs_tab()
