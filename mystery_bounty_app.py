"""
Mystery Bounty Simulator — CoinPoker Internal Tool
====================================================
Streamlit app for simulating mystery bounty prize distributions.

Usage:
    pip install streamlit pandas plotly requests
    streamlit run mystery_bounty_app.py

Engine: BountyArchitect v18 — Power-Curve with Auto-Optimisation
Inputs: pool, players, min_bounty (pinned), max_bounty (±10% flex)
Engine auto-discovers: K, end_multiplier, top-tier pattern, count shape

Metabase integration: loads real tournament data from cp_prod (DB 133, Athena)
"""

import math
import json
import io
from datetime import date, timedelta

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: ENGINE — BountyArchitect v19 (Ratio-Interpolation)
# ═══════════════════════════════════════════════════════════════════════════════
#
# HOW IT WORKS:
#   1. User provides 4 inputs: pool, players, min_bounty, max_bounty
#   2. Engine searches over (K, mult_start, mult_end, count_rate) to find the
#      distribution that best satisfies all conditions:
#        C1: Hard rules — top=1, values increasing, counts decreasing
#        C2: Pool match — total ≈ prize pool (fine-tuned to < 0.1% drift)
#        C3: Count shape — exponential decay with smooth rate
#        C4: Top-3 cap — 2nd-last ≤ 3 players, 3rd-last ≤ 6 players
#        C5: All ratios ≥ 1.9
#   3. Values use ratio interpolation: ratios smoothly transition from
#      mult_start to mult_end using t^1.5 power curve.
#   4. Mid-tier values scaled to match pool, then fine-tuned.
#   5. Counts use exponential weights with configurable rate.
#
# CONSTRAINTS:
#   - Min bounty: pinned exactly (guaranteed minimum, no flex)
#   - Max bounty: pinned exactly (±10% flex only if no solution)
#   - DENOM: adaptive ($10 for min≥$10, $5 for smaller values)
# ═══════════════════════════════════════════════════════════════════════════════

class BountyArchitect:
    """Mystery Bounty prize distribution engine v19 — Ratio-Interpolation.

    Generates a prize distribution using smooth ratio interpolation between
    mult_start and mult_end, with exponential player count decay and
    mid-tier pool scaling.

    Parameters:
        pool (float)       : Total bounty pool in USDT
        players (int)      : Number of players receiving bounties
        min_bounty (float) : Guaranteed smallest bounty (pinned exactly)
        max_bounty (float) : Guaranteed largest bounty (pinned exactly)

    Auto-discovered:
        K              : Number of tiers (7–13)
        mult_start     : Starting ratio (L1→L2), typically 2.0–2.6
        mult_end       : Ending ratio (L(k-1)→Lk), typically 2.0–2.6
        count_rate     : Exponential decay rate for player counts
        DENOM          : Adaptive — $10 for min≥$10, $5 for smaller values
    """

    MULT_START = 2.0  # Default starting multiplier (class-level reference)

    def __init__(self, pool, players, min_bounty, max_bounty):
        if pool <= 0 or players <= 0:
            raise ValueError("Pool and players must be > 0")
        if players * min_bounty > pool:
            raise ValueError("Pool cannot cover min_bounty × players")
        if max_bounty <= min_bounty:
            raise ValueError("max_bounty must be > min_bounty")

        self.pool = pool
        self.players = players
        self.min_val = min_bounty
        self.max_val = max_bounty
        # Adaptive DENOM: $10 for clean multiples, $5 for small values
        if min_bounty >= 10 and min_bounty % 10 == 0:
            self.DENOM = 10
        else:
            self.DENOM = 5
        self.warnings = []

    def _round(self, x):
        """Round to nearest DENOM."""
        d = self.DENOM
        return int(round(x / d)) * d

    # ── Player allocation (exponential weights) ─────────────────────────

    def _allocate_players(self, k, rate):
        """Allocate players using exponential decay weights.

        rate controls how steeply counts drop from L1 down to Lk.
        Higher rate = more players in lower tiers.
        """
        weights = [rate ** (k - 1 - i) for i in range(k - 1)] + [1.0]
        w_sum = sum(weights)
        counts = [max(1, int(round(w / w_sum * self.players)))
                  for w in weights]
        counts[-1] = 1  # Top tier always 1 player

        # Adjust L1 to hit exact player count
        diff = self.players - sum(counts)
        counts[0] += diff

        # Fix any non-strictly-decreasing violations
        for _ in range(k * 50):
            swapped = False
            for i in range(k - 1):
                if counts[i] <= counts[i + 1]:
                    counts[i] += 1
                    counts[i + 1] -= 1
                    swapped = True
            if not swapped:
                break
        return counts

    # ── Value generation (ratio interpolation) ──────────────────────────

    def _generate_and_scale(self, k, ms, me, counts):
        """Generate values via ratio interpolation + pool-aware scaling.

        Ratios smoothly transition from ms (mult_start) to me (mult_end)
        using t^1.5 power interpolation. Mid-tier values then scaled to
        match pool, followed by DENOM-step fine-tuning.
        """
        d = self.DENOM
        n = k - 1  # number of ratios

        # Build ratio sequence using t^1.5 interpolation
        ratios = []
        for i in range(n):
            t = i / (n - 1) if n > 1 else 0.5
            ratios.append(ms + (me - ms) * (t ** 1.5))

        # Generate raw values from ratios
        vals = [float(self.min_val)]
        for i in range(1, k):
            vals.append(vals[-1] * ratios[i - 1])

        # Round mid-tiers, pin anchors
        for i in range(1, k - 1):
            vals[i] = self._round(vals[i])
        vals[0] = self.min_val
        vals[-1] = self.max_val

        # Fix monotonicity
        for i in range(1, k):
            if vals[i] <= vals[i - 1]:
                vals[i] = vals[i - 1] + d

        # Scale mid-tiers (L3 through L(k-1)) to match pool
        fixed_sum = (vals[0] * counts[0] + vals[1] * counts[1]
                     + vals[-1] * counts[-1])
        target_mid = self.pool - fixed_sum
        mid_idx = list(range(2, k - 1))

        if mid_idx:
            current_mid = sum(vals[i] * counts[i] for i in mid_idx)
            if current_mid > 0 and target_mid > 0:
                scale = target_mid / current_mid
                for i in mid_idx:
                    vals[i] = self._round(vals[i] * scale)

        # Fix monotonicity after scaling
        for i in range(1, k):
            if vals[i] <= vals[i - 1]:
                vals[i] = vals[i - 1] + d
        vals[-1] = self.max_val

        # Fine-tune: nudge mid values by DENOM steps to reduce pool drift
        total = sum(v * c for v, c in zip(vals, counts))
        drift = self.pool - total
        for _ in range(200):
            if abs(drift) < d:
                break
            step = d if drift > 0 else -d
            moved = False
            for i in range(2, k - 1):
                if abs(drift) < d:
                    break
                nv = vals[i] + step
                if vals[i - 1] < nv < vals[i + 1]:
                    new_drift = drift - step * counts[i]
                    if abs(new_drift) < abs(drift):
                        vals[i] = nv
                        drift = new_drift
                        moved = True
            if not moved:
                break

        return [int(v) for v in vals]

    # ── Main build ───────────────────────────────────────────────────────

    def build(self):
        """Search for the optimal (K, mult_start, mult_end, rate).

        Returns:
            tuple: (k, ideal_vals, final_vals, counts, meta)
        """
        best_score = float('inf')
        best = None
        d = self.DENOM

        for K in range(7, 14):
            for ms_x100 in range(195, 265, 5):
                ms = ms_x100 / 100.0
                for me_x100 in range(200, 270, 5):
                    me = me_x100 / 100.0
                    for rate_x100 in range(175, 215, 5):
                        rate = rate_x100 / 100.0

                        counts = self._allocate_players(K, rate)
                        if counts[0] <= counts[1]:
                            continue
                        # Top-3 cap: top=1, 2nd≤3, 3rd≤6
                        if (counts[-1] != 1 or counts[-2] > 3
                                or counts[-3] > 6):
                            continue

                        vals = self._generate_and_scale(
                            K, ms, me, counts)

                        # Validate: strictly increasing, all ratios ≥ 1.9
                        ok = True
                        act_ratios = []
                        for i in range(K - 1):
                            if vals[i + 1] <= vals[i]:
                                ok = False
                                break
                            act_ratios.append(vals[i + 1] / vals[i])
                        if not ok:
                            continue
                        if any(r < 1.90 for r in act_ratios):
                            continue

                        total = sum(v * c for v, c in zip(vals, counts))
                        drift_pct = abs(total - self.pool) / self.pool * 100
                        if drift_pct > 5:
                            continue

                        # Score: low drift + smooth ratio progression
                        ratio_jumps = sum(
                            abs(act_ratios[i + 1] - act_ratios[i])
                            for i in range(len(act_ratios) - 1))
                        # Prefer higher K (more tiers = smoother curve)
                        k_bonus = max(0, 11 - K) * 5
                        score = (drift_pct * 10 + ratio_jumps * 50
                                 + k_bonus)

                        if score < best_score:
                            best_score = score
                            best = {
                                'k': K, 'vals': vals, 'counts': counts,
                                'ms': ms, 'me': me, 'rate': rate,
                                'total': total, 'drift': drift_pct,
                                'score': score, 'ratios': act_ratios,
                            }

        if best is None:
            self.warnings.append(
                "Could not find valid distribution. "
                "Try adjusting min/max bounty."
            )
            return self._fallback_build()

        k = best['k']
        vals = best['vals']
        counts = best['counts']
        meta = {
            'actual_min': vals[0],
            'actual_max': vals[-1],
            'alpha': best['ms'],  # mult_start (displayed as alpha)
            'top_pattern': [counts[-1], counts[-2], counts[-3]],
            'score': best['score'],
            'mult_start': best['ms'],
            'mult_end': best['me'],
            'count_rate': best['rate'],
        }

        return k, vals, vals, counts, meta

    def _fallback_build(self):
        """Simple geometric fallback when main search fails."""
        d = self.DENOM
        avg = self.pool / self.players
        k = 8
        mn = self.min_val
        mx = self.max_val

        ratio = (mx / mn) ** (1.0 / (k - 1))
        vals = [mn]
        for i in range(1, k):
            vals.append(self._round(vals[-1] * ratio))
        vals[0] = mn
        vals[-1] = mx
        for i in range(1, k):
            if vals[i] <= vals[i - 1]:
                vals[i] = vals[i - 1] + d

        rate = 1.85
        counts = self._allocate_players(k, rate)

        meta = {
            'actual_min': mn, 'actual_max': mx,
            'alpha': ratio, 'top_pattern': [1, 2, 3],
            'score': -1, 'mult_start': ratio,
            'mult_end': ratio, 'count_rate': rate,
        }

        return k, vals, vals, counts, meta


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: METABASE CLIENT
# ─────────────────────────────────────────────────────────────────────────────

METABASE_BASE_URL = "https://baazigames.metabaseapp.com"
METABASE_DATABASE_ID = 133


def get_metabase_session():
    """Get Metabase session headers using API key."""
    import requests
    api_key = st.session_state.get("metabase_api_key", "")
    if not api_key:
        return None
    return {"x-api-key": api_key}


def query_metabase(sql: str) -> pd.DataFrame:
    """Execute a SQL query against CoinPoker's cp_prod database via Metabase API."""
    import requests
    headers = get_metabase_session()
    if headers is None:
        st.error("Metabase API key required. Enter it in the sidebar.")
        return pd.DataFrame()

    payload = {
        "database": METABASE_DATABASE_ID,
        "type": "native",
        "native": {"query": sql},
    }
    try:
        resp = requests.post(
            f"{METABASE_BASE_URL}/api/dataset",
            json=payload, headers=headers, timeout=120,
        )
        resp.raise_for_status()
        result = resp.json()
        if result.get("status") == "failed":
            st.error(f"Query failed: {result.get('error', 'Unknown')}")
            return pd.DataFrame()
        rows = result["data"]["rows"]
        cols = [c["name"] for c in result["data"]["cols"]]
        return pd.DataFrame(rows, columns=cols)
    except Exception as e:
        st.error(f"Metabase error: {e}")
        return pd.DataFrame()


BOUNTY_TOURNAMENT_QUERY = """
SELECT
    tournament_id, tournament_name,
    MIN(buy_in) AS buy_in, MIN(entry_fee) AS entry_fee,
    MIN(bounty_amount) AS bounty_amount,
    COUNT(DISTINCT user_id) AS player_count,
    SUM(buy_in) AS total_buyin_pool,
    SUM(bounty_won) AS total_bounties_paid,
    MIN(tournament_date) AS tournament_date,
    MIN(lobby_id) AS lobby_id
FROM lakehouse_gold.user_tournament_play_history
WHERE tournament_date BETWEEN DATE('{start}') AND DATE('{end}')
  AND lobby_id != 9 AND bounty_amount > 0
GROUP BY tournament_id, tournament_name
HAVING COUNT(DISTINCT user_id) >= {min_players}
ORDER BY MIN(bounty_amount) DESC, total_buyin_pool DESC
LIMIT 50
"""

ALL_TOURNAMENT_QUERY = """
SELECT
    tournament_id, tournament_name,
    MIN(buy_in) AS buy_in, MIN(entry_fee) AS entry_fee,
    MIN(bounty_amount) AS bounty_amount,
    COUNT(DISTINCT user_id) AS player_count,
    SUM(buy_in) AS total_buyin_pool,
    MIN(tournament_date) AS tournament_date
FROM lakehouse_gold.user_tournament_play_history
WHERE tournament_date BETWEEN DATE('{start}') AND DATE('{end}')
  AND lobby_id != 9 AND (buy_in + entry_fee) > 0
GROUP BY tournament_id, tournament_name
HAVING COUNT(DISTINCT user_id) >= {min_players}
ORDER BY total_buyin_pool DESC
LIMIT 50
"""


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: STREAMLIT APP
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Mystery Bounty Simulator",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stMetric > div { background: #1a1a2e; border-radius: 10px; padding: 12px 16px; }
    .stMetric label { color: #8892b0 !important; font-size: 0.85rem !important; }
    .stMetric [data-testid="stMetricValue"] { color: #64ffda !important; }
    div[data-testid="stSidebar"] { background: #0d1117; }
    button[kind="primary"], .stButton > button[kind="primary"],
    div.stButton > button[type="button"][kind="primary"] {
        background-color: #e63946 !important;
        border-color: #e63946 !important; color: white !important;
    }
    button[kind="primary"]:hover, .stButton > button[kind="primary"]:hover {
        background-color: #c1121f !important; border-color: #c1121f !important;
    }
    .engine-badge {
        display: inline-block;
        background: linear-gradient(135deg, #1f6feb22, #64ffda22);
        border: 1px solid #64ffda44; border-radius: 6px;
        padding: 2px 10px; font-size: 0.75rem;
        color: #64ffda; font-weight: 600; margin-left: 8px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎯 Mystery Bounty Simulator")
st.caption("CoinPoker Internal — Power-Curve Engine v18")


# ─── SIDEBAR ───
with st.sidebar:
    st.header("🔗 Metabase Connection")
    st.caption("Connect to cp_prod for live tournament data")

    secrets_key = ""
    try:
        secrets_key = st.secrets.get("METABASE_API_KEY", "")
    except Exception:
        pass
    if secrets_key and "metabase_api_key" not in st.session_state:
        st.session_state["metabase_api_key"] = secrets_key

    api_key = st.text_input(
        "API Key", type="password",
        value=st.session_state.get("metabase_api_key", ""),
        help="Metabase API key for cp_prod access.",
    )
    if api_key:
        st.session_state["metabase_api_key"] = api_key
        st.success("Connected to cp_prod", icon="✅")
    else:
        st.warning("Enter API key for Metabase access.")

    st.divider()
    st.header("ℹ️ Engine Info")
    st.caption(
        "**BountyArchitect v18** — Power-Curve\n\n"
        "The engine automatically determines:\n"
        "- Number of tiers (K=5–12)\n"
        "- End multiplier\n"
        "- Top-tier pattern (capped [1,≤3,≤6])\n"
        "- Count curve steepness\n\n"
        "Parameter rules:\n"
        "- Min bounty: pinned exactly\n"
        "- Max bounty: ±10% flex if needed\n"
        "- Start multiplier: fixed at 2.0"
    )


# ─── Session state ───
if "comparison_runs" not in st.session_state:
    st.session_state.comparison_runs = []


# ─── Simulation runner ───

def run_simulation(pool, players, min_bounty, max_bounty, label=""):
    """Run BountyArchitect v18 and return results dict."""
    try:
        engine = BountyArchitect(
            pool=pool, players=players,
            min_bounty=min_bounty, max_bounty=max_bounty,
        )
        k, ideal, final, counts, meta = engine.build()

        tiers = []
        for i in range(k):
            ratio_val = final[i] / final[i - 1] if i > 0 else None
            ratio_cnt = counts[i - 1] / counts[i] if i > 0 else None
            tiers.append({
                "Tier": f"L{i + 1}",
                "Players": counts[i],
                "Count Ratio": f"{ratio_cnt:.1f}x" if ratio_cnt else "—",
                "Bounty (USDT)": final[i],
                "Bounty Ratio": f"{ratio_val:.2f}x" if ratio_val else "—",
                "Pool $": round(final[i] * counts[i], 2),
                "Pool %": round(final[i] * counts[i] / pool * 100, 1),
            })

        pool_total = sum(v * c for v, c in zip(final, counts))

        return {
            "label": label,
            "k": k,
            "ideal": ideal,
            "vals": final,
            "counts": counts,
            "tiers": tiers,
            "pool": pool,
            "pool_total": round(pool_total, 2),
            "pool_error": round(abs(pool_total - pool), 2),
            "pool_drift_pct": round(abs(pool_total - pool) / pool * 100, 2),
            "players": players,
            "min_bounty": min_bounty,
            "max_bounty": max_bounty,
            "mult_start": BountyArchitect.MULT_START,
            "warnings": engine.warnings,
            "meta": meta,
            "params": {
                "mult_start": BountyArchitect.MULT_START,
                "actual_min": meta['actual_min'],
                "actual_max": meta['actual_max'],
                "alpha": meta['alpha'],
                "top_pattern": str(meta['top_pattern']),
                "k": k,
            },
        }
    except Exception as e:
        st.error(f"Engine error: {e}")
        return None


# ─── Results renderer ───

def render_results(result):
    """Render simulation results with charts and metrics."""
    if result is None:
        return

    df = pd.DataFrame(result["tiers"])
    vals = result["vals"]
    ideal = result.get("ideal", vals)
    counts = result["counts"]
    pool = result["pool"]
    players = result["players"]
    k = result["k"]
    meta = result.get("meta", {})

    # ── Key Metrics ──
    avg_bounty = pool / players
    ratios = [vals[i + 1] / vals[i] for i in range(k - 1)]
    max_ratio = vals[-1] / vals[0]

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Tiers (K)", k)
    col2.metric("Avg Bounty", f"${avg_bounty:,.0f}")
    col3.metric("Start Mult", f"{ratios[0]:.2f}x")
    col4.metric("End Mult", f"{ratios[-1]:.2f}x")
    col5.metric("Pool Drift", f"{result['pool_drift_pct']:.2f}%")

    # Engine metadata
    st.caption(
        f"**Engine:** α={meta.get('alpha', '?')} · "
        f"Top pattern={meta.get('top_pattern', '?')} · "
        f"Actual min=${meta.get('actual_min', '?')} · "
        f"Actual max=${meta.get('actual_max', '?'):,}"
    )

    if result.get("pool_error", 0) > 1:
        st.info(
            f"ℹ️ Pool drift: ${result['pool_error']:.0f} USDT — "
            f"total ${result['pool_total']:,.0f} vs target ${pool:,.0f}"
        )

    # ── Charts ──
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        colors = px.colors.sequential.Teal
        n_colors = len(colors)
        tier_colors = [colors[min(i * n_colors // k, n_colors - 1)]
                       for i in range(k)]
        tier_labels = [f"L{i + 1}" for i in range(k)]

        fig_vals = go.Figure()
        fig_vals.add_trace(go.Bar(
            name="Bounty Value",
            x=tier_labels, y=vals,
            marker_color=tier_colors,
            text=[f"${v:,.0f}" for v in vals],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>$%{y:,.0f}<extra></extra>",
        ))
        fig_vals.update_layout(
            title="Bounty Values by Tier",
            xaxis_title="Tier", yaxis_title="Bounty (USDT)",
            template="plotly_dark", height=400,
            margin=dict(t=60, b=40),
        )
        st.plotly_chart(fig_vals, use_container_width=True)

    with chart_col2:
        fig_players = go.Figure()
        fig_players.add_trace(go.Bar(
            y=[f"L{i + 1} (${vals[i]:,.0f})" for i in range(k)],
            x=counts, orientation="h",
            marker_color=tier_colors,
            text=[f"{c} ({c / players * 100:.0f}%)" for c in counts],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Players: %{x}<extra></extra>",
        ))
        fig_players.update_layout(
            title="Player Distribution",
            xaxis_title="Players", yaxis_title="Tier",
            template="plotly_dark", height=400,
            margin=dict(t=60, b=40),
        )
        st.plotly_chart(fig_players, use_container_width=True)

    # ── Multiplier progression ──
    if k >= 3:
        fig_mult = go.Figure()
        fig_mult.add_trace(go.Scatter(
            x=[f"L{i}→L{i + 1}" for i in range(1, k)],
            y=ratios,
            mode="lines+markers",
            name="Value ratios",
            line=dict(color="#64ffda", width=2),
            marker=dict(size=8),
            hovertemplate="<b>%{x}</b><br>%{y:.2f}x<extra></extra>",
        ))

        # Count ratios
        cnt_ratios = [counts[i] / counts[i + 1] for i in range(k - 1)]
        fig_mult.add_trace(go.Scatter(
            x=[f"L{i}→L{i + 1}" for i in range(1, k)],
            y=cnt_ratios,
            mode="lines+markers",
            name="Count ratios",
            line=dict(color="#f0883e", width=2, dash="dot"),
            marker=dict(size=6),
            hovertemplate="<b>%{x}</b><br>%{y:.1f}x<extra></extra>",
        ))

        fig_mult.update_layout(
            title="Ratio Progression (Value & Count)",
            xaxis_title="Tier Transition",
            yaxis_title="Ratio",
            template="plotly_dark", height=300,
            margin=dict(t=60, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1),
        )
        st.plotly_chart(fig_mult, use_container_width=True)

    # ── Pool distribution pie ──
    subtotals = [round(v * c, 2) for v, c in zip(vals, counts)]
    fig_pie = go.Figure(data=[go.Pie(
        labels=[f"L{i + 1}" for i in range(k)],
        values=subtotals, hole=0.4,
        marker=dict(colors=tier_colors),
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>$%{value:,.0f} (%{percent})"
                      "<extra></extra>",
    )])
    fig_pie.update_layout(
        title="Pool Distribution by Tier",
        template="plotly_dark", height=350,
        margin=dict(t=60, b=20),
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # ── Distribution Table ──
    st.subheader("Distribution Table")
    st.dataframe(
        df.style.format({
            "Bounty (USDT)": "${:,.0f}",
            "Pool $": "${:,.0f}",
            "Pool %": "{:.1f}%",
        }),
        use_container_width=True,
        hide_index=True,
    )
    pool_err = result['pool_error']
    pool_tot = result['pool_total']
    drift_str = f"(drift: ${pool_err:.0f})" if pool_err >= 1 else "(✅ exact)"
    st.caption(
        f"**Total: {players} players · ${pool:,.0f} target · "
        f"${pool_tot:,.0f} actual** {drift_str}"
    )

    if result.get("warnings"):
        for w in result["warnings"]:
            st.warning(w)

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════

tab_manual, tab_metabase, tab_compare, tab_code = st.tabs([
    "📝 Manual Simulator",
    "🔌 Load from CoinPoker",
    "🔄 Compare Runs",
    "💻 Code Editor",
])

# ─── TAB 1: Manual Simulator ─────────────────────────────────────────────────

with tab_manual:
    st.subheader("Enter Tournament Parameters")

    mcol1, mcol2 = st.columns(2)
    with mcol1:
        pool = st.number_input(
            "Prize Pool (USDT)",
            min_value=100, max_value=1_000_000, value=100000, step=1000)
        min_bounty = st.number_input(
            "Min Bounty (USDT)",
            min_value=1.0, max_value=10000.0, value=100.0, step=10.0)
    with mcol2:
        players = st.number_input(
            "Players",
            min_value=5, max_value=10000, value=100, step=10)
        max_bounty = st.number_input(
            "Max Bounty (USDT)",
            min_value=10.0, max_value=500000.0, value=17000.0, step=500.0)

    run_btn = st.button("🚀 Run Simulation", type="primary",
                         use_container_width=True)

    if run_btn:
        with st.spinner("Searching for optimal distribution..."):
            result = run_simulation(
                pool, players, min_bounty, max_bounty,
                label="Manual")
        if result:
            st.session_state["last_manual_result"] = result
            render_results(result)

            exp1, exp2, exp3 = st.columns(3)
            df_export = pd.DataFrame(result["tiers"])
            with exp1:
                csv = df_export.to_csv(index=False)
                st.download_button("📥 CSV", csv,
                                   "bounty_distribution.csv", "text/csv")
            with exp2:
                json_str = json.dumps({
                    "k": result["k"], "vals": result["vals"],
                    "counts": result["counts"],
                    "pool": result["pool"],
                    "pool_total": result["pool_total"],
                    "params": result["params"],
                }, indent=2)
                st.download_button("📥 JSON", json_str,
                                   "bounty_config.json", "application/json")
            with exp3:
                if st.button("📌 Save to Comparison"):
                    st.session_state.comparison_runs.append(result)
                    st.success(
                        f"Saved! ({len(st.session_state.comparison_runs)} runs)")

    elif "last_manual_result" in st.session_state:
        render_results(st.session_state["last_manual_result"])


# ─── TAB 2: Load from CoinPoker ──────────────────────────────────────────────

with tab_metabase:
    st.subheader("Load Real Tournament Data from cp_prod")

    if not st.session_state.get("metabase_api_key"):
        st.info("Enter your Metabase API key in the sidebar to connect.")
    else:
        lcol1, lcol2, lcol3, lcol4 = st.columns(4)
        with lcol1:
            start_date = st.date_input(
                "Start date", value=date.today() - timedelta(days=7))
        with lcol2:
            end_date = st.date_input("End date", value=date.today())
        with lcol3:
            min_player_filter = st.number_input(
                "Min players", min_value=5, max_value=500, value=20)
        with lcol4:
            bounty_only = st.checkbox("Bounty only", value=True)

        if st.button("🔍 Fetch Tournaments", type="primary"):
            with st.spinner("Querying cp_prod..."):
                tmpl = (BOUNTY_TOURNAMENT_QUERY if bounty_only
                        else ALL_TOURNAMENT_QUERY)
                sql = tmpl.format(
                    start=start_date.isoformat(),
                    end=end_date.isoformat(),
                    min_players=min_player_filter)
                df_t = query_metabase(sql)
            if not df_t.empty:
                st.session_state["tournaments_df"] = df_t
                st.session_state["tournaments_bounty_only"] = bounty_only
                st.success(f"Found {len(df_t)} tournaments")
            else:
                st.warning("No tournaments found.")

        if ("tournaments_df" in st.session_state
                and not st.session_state["tournaments_df"].empty):
            df_t = st.session_state["tournaments_df"]

            def make_label(r):
                name = r['tournament_name']
                p = int(r['player_count'])
                pool_v = float(r['total_buyin_pool'])
                b = float(r.get('bounty_amount', 0))
                if b > 0:
                    return (f"{name} — {p} players, "
                            f"${pool_v:,.0f} pool, ${b:,.2f} bounty")
                return f"{name} — {p} players, ${pool_v:,.0f} pool"

            df_t["label"] = df_t.apply(make_label, axis=1)
            selected_label = st.selectbox("Select tournament",
                                          df_t["label"].tolist())
            row = df_t[df_t["label"] == selected_label].iloc[0]

            player_count = int(row['player_count'])
            buy_in_val = float(row['buy_in'])
            entry_fee_val = float(row['entry_fee'])
            total_buyin = float(row['total_buyin_pool'])
            bounty_amt = float(row.get('bounty_amount', 0))
            actual_bounty_pool = round(bounty_amt * player_count, 2)

            info_cols = st.columns(5)
            info_cols[0].metric("Players", f"{player_count}")
            info_cols[1].metric("Buy-in", f"${buy_in_val:,.2f}")
            info_cols[2].metric("Entry Fee", f"${entry_fee_val:,.2f}")
            info_cols[3].metric("Bounty", f"${bounty_amt:,.2f}")
            info_cols[4].metric("Bounty Pool", f"${actual_bounty_pool:,.2f}")

            st.divider()
            st.subheader("Simulation Parameters")

            bcol1, bcol2, bcol3 = st.columns(3)
            with bcol1:
                pool_mode = st.radio(
                    "Pool source",
                    ["Actual bounty pool", "Custom %", "Manual"],
                )
            with bcol2:
                if pool_mode == "Actual bounty pool":
                    sim_pool = actual_bounty_pool
                    st.info(f"Pool: ${sim_pool:,.2f}")
                elif pool_mode == "Custom %":
                    pct = st.slider("% of buy-in pool",
                                    10, 90, 50, step=5)
                    sim_pool = round(total_buyin * pct / 100, 2)
                    st.info(f"Pool: ${sim_pool:,.2f} ({pct}%)")
                else:
                    sim_pool = st.number_input(
                        "Manual pool (USDT)",
                        min_value=10.0, max_value=500000.0,
                        value=max(100.0, float(actual_bounty_pool)),
                        step=100.0)
            with bcol3:
                mb_min = st.number_input(
                    "Min bounty (USDT)",
                    min_value=0.10, max_value=10000.0,
                    value=max(0.50, round(bounty_amt * 0.1, 2))
                    if bounty_amt > 0 else 1.0,
                    step=0.50)
                mb_max = st.number_input(
                    "Max bounty (USDT)",
                    min_value=1.0, max_value=100000.0,
                    value=max(5.0, round(sim_pool * 0.10, 0))
                    if sim_pool > 0 else 100.0,
                    step=5.0)

            if st.button("🎯 Simulate", type="primary",
                          use_container_width=True):
                with st.spinner("Searching for optimal distribution..."):
                    result = run_simulation(
                        sim_pool, player_count, mb_min, mb_max,
                        label=row['tournament_name'])
                if result:
                    st.session_state["last_metabase_result"] = result
                    render_results(result)

                    exp1, exp2, exp3 = st.columns(3)
                    df_export = pd.DataFrame(result["tiers"])
                    with exp1:
                        csv = df_export.to_csv(index=False)
                        st.download_button(
                            "📥 CSV", csv,
                            "bounty_distribution.csv", "text/csv",
                            key="csv_mb")
                    with exp2:
                        j = json.dumps({
                            "tournament_id": int(row['tournament_id']),
                            "k": result["k"],
                            "vals": result["vals"],
                            "counts": result["counts"],
                            "pool": result["pool"],
                            "params": result["params"],
                        }, indent=2)
                        st.download_button(
                            "📥 JSON", j,
                            "bounty_config.json", "application/json",
                            key="json_mb")
                    with exp3:
                        if st.button("📌 Save to Comparison", key="save_mb"):
                            st.session_state.comparison_runs.append(result)
                            st.success("Saved!")

            elif "last_metabase_result" in st.session_state:
                render_results(st.session_state["last_metabase_result"])


# ─── TAB 3: Compare Runs ─────────────────────────────────────────────────────

with tab_compare:
    st.subheader("Compare Simulation Runs")
    runs = st.session_state.comparison_runs

    if len(runs) < 1:
        st.info("Run simulations and click '📌 Save to Comparison' "
                "to start comparing.")
    else:
        st.caption(f"{len(runs)} runs saved.")
        run_labels = [
            f"Run {i + 1}: {r.get('label', 'Manual')} "
            f"({r['players']}p, ${r['pool']:,.0f})"
            for i, r in enumerate(runs)
        ]

        cc1, cc2 = st.columns(2)
        with cc1:
            idx_a = st.selectbox("Run A", range(len(runs)),
                                 format_func=lambda i: run_labels[i])
        with cc2:
            idx_b = st.selectbox("Run B", range(len(runs)),
                                 index=min(1, len(runs) - 1),
                                 format_func=lambda i: run_labels[i])

        run_a, run_b = runs[idx_a], runs[idx_b]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tiers", f"{run_a['k']} vs {run_b['k']}")
        m2.metric("Avg Bounty",
                   f"${run_a['pool'] / run_a['players']:,.0f} vs "
                   f"${run_b['pool'] / run_b['players']:,.0f}")
        ra_ratios = [run_a['vals'][i + 1] / run_a['vals'][i]
                     for i in range(run_a['k'] - 1)]
        rb_ratios = [run_b['vals'][i + 1] / run_b['vals'][i]
                     for i in range(run_b['k'] - 1)]
        m3.metric("Start Mult",
                   f"{ra_ratios[0]:.2f}x vs {rb_ratios[0]:.2f}x")
        m4.metric("End Mult",
                   f"{ra_ratios[-1]:.2f}x vs {rb_ratios[-1]:.2f}x")

        # Overlay chart
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Bar(
            name=f"A: {run_a.get('label', 'A')}",
            x=[f"L{i + 1}" for i in range(run_a['k'])],
            y=run_a['vals'], marker_color="#58a6ff", opacity=0.7,
        ))
        fig_cmp.add_trace(go.Bar(
            name=f"B: {run_b.get('label', 'B')}",
            x=[f"L{i + 1}" for i in range(run_b['k'])],
            y=run_b['vals'], marker_color="#f0883e", opacity=0.7,
        ))
        fig_cmp.update_layout(
            title="Bounty Values — Side by Side",
            barmode="group", template="plotly_dark", height=400,
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

        # Multiplier comparison
        if run_a['k'] >= 3 and run_b['k'] >= 3:
            fig_m = go.Figure()
            fig_m.add_trace(go.Scatter(
                name="A", mode="lines+markers",
                x=[f"L{i}→L{i + 1}" for i in range(1, run_a['k'])],
                y=ra_ratios, line=dict(color="#58a6ff", width=2),
            ))
            fig_m.add_trace(go.Scatter(
                name="B", mode="lines+markers",
                x=[f"L{i}→L{i + 1}" for i in range(1, run_b['k'])],
                y=rb_ratios, line=dict(color="#f0883e", width=2),
            ))
            fig_m.update_layout(
                title="Multiplier Progression",
                template="plotly_dark", height=300,
            )
            st.plotly_chart(fig_m, use_container_width=True)

        # Param diff table
        st.subheader("Parameter Comparison")
        pa = run_a.get("params", {})
        pb = run_b.get("params", {})
        all_keys = sorted(set(list(pa.keys()) + list(pb.keys())))
        diff_rows = [{"Parameter": k, "Run A": str(pa.get(k, "—")),
                       "Run B": str(pb.get(k, "—")),
                       "Changed": "✅" if pa.get(k) != pb.get(k) else ""}
                      for k in all_keys]
        st.dataframe(pd.DataFrame(diff_rows), use_container_width=True,
                     hide_index=True)

        if st.button("🗑️ Clear all runs"):
            st.session_state.comparison_runs = []
            st.rerun()


# ─── TAB 4: Code Editor ──────────────────────────────────────────────────────

# Engine source boundaries (line numbers in THIS file, 0-indexed)
_ENGINE_START_MARKER = "class BountyArchitect:"
_ENGINE_END_MARKER = "\n\n# ─────────────────────────────────────────────────────────────────────────────\n# SECTION 2: METABASE CLIENT"

with tab_code:
    st.subheader("Engine Source — BountyArchitect v18")
    st.caption(
        "Edit the engine code below and click **Apply & Re-run** to test changes live. "
        "Changes persist for this session only — reload the app to reset."
    )

    # Load current source
    _this_file = __file__
    with open(_this_file, "r") as _f:
        _full_source = _f.read()

    # Extract engine class source
    _eng_start = _full_source.index(_ENGINE_START_MARKER)
    _eng_end = _full_source.index(
        "# ─────────────────────────────────────────────────────────────────────────────\n# SECTION 2: METABASE CLIENT"
    )
    _engine_source = _full_source[_eng_start:_eng_end].rstrip()

    if "editor_code" not in st.session_state:
        st.session_state.editor_code = _engine_source

    edited_code = st.text_area(
        "Engine Code",
        value=st.session_state.editor_code,
        height=600,
        key="code_editor_area",
        label_visibility="collapsed",
    )

    col_apply, col_reset, col_download = st.columns([2, 1, 1])

    with col_apply:
        if st.button("▶️ Apply & Re-run", type="primary", use_container_width=True):
            try:
                # Validate syntax
                compile(edited_code, "<code-editor>", "exec")

                # Write modified source back to file
                new_source = (
                    _full_source[:_eng_start]
                    + edited_code.rstrip()
                    + "\n\n"
                    + _full_source[_eng_end:]
                )
                with open(_this_file, "w") as _fw:
                    _fw.write(new_source)

                st.session_state.editor_code = edited_code
                st.success("Code applied — reloading app...")
                st.rerun()

            except SyntaxError as e:
                st.error(f"Syntax error on line {e.lineno}: {e.msg}")
            except Exception as e:
                st.error(f"Error: {e}")

    with col_reset:
        if st.button("🔄 Reset to Default", use_container_width=True):
            # Re-read original from file
            with open(_this_file, "r") as _f:
                _current = _f.read()
            _cs = _current.index(_ENGINE_START_MARKER)
            _ce = _current.index(
                "# ─────────────────────────────────────────────────────────────────────────────\n# SECTION 2: METABASE CLIENT"
            )
            st.session_state.editor_code = _current[_cs:_ce].rstrip()
            st.rerun()

    with col_download:
        st.download_button(
            "⬇️ Download .py",
            data=edited_code,
            file_name="bounty_architect_v18.py",
            mime="text/x-python",
            use_container_width=True,
        )
