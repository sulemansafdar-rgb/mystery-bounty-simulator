"""
Mystery Bounty Simulator — CoinPoker Internal Tool
====================================================
Streamlit app for simulating and fine-tuning mystery bounty prize distributions.

Usage:
    pip install streamlit pandas plotly requests
    streamlit run mystery_bounty_app.py

Metabase integration: loads real tournament data from cp_prod (DB 133, Athena)
to simulate mystery bounty distributions on actual CoinPoker tournaments.
"""

import math
import json
import io
from datetime import date, timedelta

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ─────────────────────────────────────────────────────────────────────────────
# ENGINE — BountyArchitect v6.2 (embedded for single-file deployment)
# ─────────────────────────────────────────────────────────────────────────────

class BountyArchitect:
    """Mystery Bounty prize distribution engine v6.2."""

    def __init__(self, pool, players, min_bounty, max_bounty,
                 mult_start=2.0, mult_end=1.5,
                 mode="power_decay", strength=1.5,
                 max_top_tier_count=8, denomination=10):
        if pool <= 0 or players <= 0:
            raise ValueError("Pool and players must be > 0")
        if max_bounty <= min_bounty:
            raise ValueError("max_bounty must be > min_bounty")
        if mult_start <= 1.0 or mult_end <= 1.0:
            raise ValueError("Multipliers must be > 1.0")
        if mult_end > mult_start:
            raise ValueError("mult_end must be <= mult_start")
        if max_top_tier_count < 3:
            raise ValueError("max_top_tier_count must be >= 3")
        if min_bounty * players > pool:
            raise ValueError(
                f"Pool impossible: min({min_bounty}) × players({players}) "
                f"= {min_bounty * players} > pool({pool})"
            )
        self.pool = pool
        self.players = players
        self.min_val = min_bounty
        self.max_val = max_bounty
        self.mult_start = mult_start
        self.mult_end = mult_end
        self.mode = mode
        self.strength = strength
        self.max_top = max_top_tier_count
        self.denom = denomination
        self.warnings = []

    def recommend_k(self):
        avg_r = math.sqrt(self.mult_start * self.mult_end)
        k = max(4, min(12, round(
            math.log(self.max_val / self.min_val) / math.log(avg_r)
        )))
        if self.mult_start > 2.5 and k > 7:
            k = max(4, math.floor(k * 0.85))
            self.warnings.append(f"High mult_start. Auto-reduced k to {k}.")
        return k

    def _generate_weights(self, k):
        if self.mode == "power_decay":
            decay = max(0.40, min(0.80, 0.85 - (self.strength * 0.10)))
            raw = [decay ** (k - 1 - i) for i in range(k)]
        elif self.mode == "linear_decay":
            raw = list(range(k, 0, -1))
        elif self.mode == "gaussian_mid":
            center = k / 2.0
            raw = [
                math.exp(-((i - center) ** 2) / (2 * self.strength ** 2))
                for i in range(1, k + 1)
            ]
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        s = sum(raw)
        return [w / s for w in raw]

    def _allocate_players(self, k):
        weights = self._generate_weights(k)
        counts = [1] * k
        remaining = self.players - k
        if remaining <= 0:
            return counts
        shares = [w * remaining for w in weights]
        floors = [math.floor(s) for s in shares]
        counts = [c + f for c, f in zip(counts, floors)]
        rem = remaining - sum(floors)
        fractions = [(s - math.floor(s), i) for i, s in enumerate(shares)]
        fractions.sort(key=lambda x: x[0], reverse=True)
        for _, i in fractions[:rem]:
            counts[i] += 1
        # Top-tier scarcity cap
        if k >= 4:
            top_idx = [k - 1, k - 2, k - 3]
            top_sum = sum(counts[i] for i in top_idx)
            if top_sum > self.max_top:
                top_w = [weights[i] for i in top_idx]
                top_tw = sum(top_w)
                new_top = [max(1, round((top_w[i] / top_tw) * self.max_top)) for i in range(len(top_idx))]
                while sum(new_top) > self.max_top:
                    max_i = max(range(len(new_top)), key=lambda x: new_top[x])
                    new_top[max_i] -= 1
                while sum(new_top) < self.max_top:
                    min_i = min(range(len(new_top)), key=lambda x: new_top[x])
                    new_top[min_i] += 1
                actual_freed = top_sum - sum(new_top)
                for i, idx in enumerate(top_idx):
                    counts[idx] = new_top[i]
                lower_idx = list(range(k - 3))
                lower_w = [weights[i] for i in lower_idx]
                lower_tw = sum(lower_w)
                distributed = 0
                for i in lower_idx:
                    add = round((lower_w[i] / lower_tw) * actual_freed)
                    counts[i] += add
                    distributed += add
                leftover = actual_freed - distributed
                if leftover != 0:
                    counts[lower_idx[0]] += leftover
        # Strict monotonic pyramid (swap-only)
        for _ in range(k * 20):
            changed = False
            for i in range(k - 1):
                if counts[i] < counts[i + 1]:
                    excess = math.ceil((counts[i + 1] - counts[i]) / 2)
                    if counts[i + 1] - excess >= 1:
                        counts[i] += excess
                        counts[i + 1] -= excess
                        changed = True
            if not changed:
                break
        for i in range(k):
            if counts[i] < 1:
                counts[i] = 1
        return counts

    def _reconcile_pool(self, vals, counts):
        k = len(vals)
        c_vals = [int(round(v * 100)) for v in vals]
        target_cents = int(round(self.pool * 100))
        step_cents = int(round(self.denom * 100))
        min_c = int(round(self.min_val * 100))
        max_c = int(round(self.max_val * 100))
        adjustable = list(range(1, k - 1))

        for _pass in range(2):
            current_step = step_cents if _pass == 0 else 100
            for _ in range(k * 200):
                current = sum(v * c for v, c in zip(c_vals, counts))
                diff = target_cents - current
                if diff == 0:
                    break
                direction = 1 if diff > 0 else -1
                step = current_step * direction
                best_idx = None
                best_remaining = abs(diff)
                for i in adjustable:
                    new_val = c_vals[i] + step
                    lo = c_vals[i - 1] + current_step if i > 0 else min_c
                    hi = c_vals[i + 1] - current_step if i < k - 1 else max_c
                    if new_val < min_c or new_val > max_c:
                        continue
                    if new_val < lo or new_val > hi:
                        continue
                    impact = abs(step * counts[i])
                    remaining = abs(abs(diff) - impact)
                    if remaining < best_remaining:
                        best_remaining = remaining
                        best_idx = i
                if best_idx is None:
                    break
                c_vals[best_idx] += step
            current = sum(v * c for v, c in zip(c_vals, counts))
            if current == target_cents:
                break

        # Exact close — single tier
        current = sum(v * c for v, c in zip(c_vals, counts))
        diff = target_cents - current
        if diff != 0:
            for i in adjustable:
                if counts[i] == 0:
                    continue
                if diff % counts[i] == 0:
                    adj = diff // counts[i]
                    new_val = c_vals[i] + adj
                    if min_c <= new_val <= max_c:
                        ok = True
                        if i > 0 and new_val <= c_vals[i - 1]:
                            ok = False
                        if i < k - 1 and new_val >= c_vals[i + 1]:
                            ok = False
                        if ok:
                            c_vals[i] = new_val
                            diff = 0
                            break

        # Two-tier exact close
        if diff != 0:
            max_delta = max(50, abs(diff) // min(c for c in counts if c > 0) + 5)
            found = False
            for a in adjustable:
                if found:
                    break
                for b in adjustable:
                    if a == b or counts[a] == 0 or counts[b] == 0:
                        continue
                    for da in range(0, max_delta):
                        remainder = diff - da * counts[a]
                        if remainder == 0:
                            continue
                        if counts[b] != 0 and (-remainder) % counts[b] == 0:
                            db = (-remainder) // counts[b]
                            if db <= 0:
                                continue
                            new_a = c_vals[a] + da
                            new_b = c_vals[b] - db
                            if (min_c <= new_a <= max_c and min_c <= new_b <= max_c
                                    and (a == 0 or new_a > c_vals[a - 1])
                                    and (a == k - 1 or new_a < c_vals[a + 1])
                                    and (b == 0 or new_b > c_vals[b - 1])
                                    and (b == k - 1 or new_b < c_vals[b + 1])):
                                c_vals[a] = new_a
                                c_vals[b] = new_b
                                diff = 0
                                found = True
                                break
                    if found:
                        break

        # Player reallocation fallback
        current = sum(v * c for v, c in zip(c_vals, counts))
        diff = target_cents - current
        if diff != 0:
            best_move = None
            best_remaining = abs(diff)
            for i in range(k - 1):
                if counts[i] <= 1:
                    continue
                delta = c_vals[i + 1] - c_vals[i]
                new_diff = diff - delta
                if abs(new_diff) < best_remaining:
                    best_remaining = abs(new_diff)
                    best_move = (i, i + 1, new_diff)
                if counts[i + 1] <= 1:
                    continue
                delta2 = c_vals[i] - c_vals[i + 1]
                new_diff2 = diff - delta2
                if abs(new_diff2) < best_remaining:
                    best_remaining = abs(new_diff2)
                    best_move = (i + 1, i, new_diff2)
            if best_move is not None and best_remaining < abs(diff):
                src, dst, new_diff = best_move
                counts[src] -= 1
                counts[dst] += 1
                diff = new_diff
                self.warnings.append(f"Moved 1 player L{src+1}→L{dst+1} for pool fit")
                if diff != 0:
                    for i in adjustable:
                        if counts[i] == 0:
                            continue
                        if diff % counts[i] == 0:
                            adj = diff // counts[i]
                            new_val = c_vals[i] + adj
                            if min_c <= new_val <= max_c:
                                ok = True
                                if i > 0 and new_val <= c_vals[i - 1]:
                                    ok = False
                                if i < k - 1 and new_val >= c_vals[i + 1]:
                                    ok = False
                                if ok:
                                    c_vals[i] = new_val
                                    diff = 0
                                    break

        final_vals = [v / 100 for v in c_vals]
        final_total = sum(v * c for v, c in zip(final_vals, counts))
        if abs(final_total - self.pool) > 0.005:
            raise ValueError(f"Pool reconciliation failed: ${final_total:.2f} != ${self.pool:.2f}")
        return final_vals

    def build(self, k=None):
        if k is None:
            k = self.recommend_k()
        multipliers = [
            self.mult_start + (self.mult_end - self.mult_start) * (i / (k - 1))
            for i in range(k - 1)
        ]
        vals = [self.min_val]
        for m in multipliers:
            vals.append(vals[-1] * m)
        vals = [round(v / self.denom) * self.denom for v in vals]
        vals[0] = self.min_val
        vals[-1] = self.max_val
        if len(vals) > 1 and vals[-2] >= vals[-1]:
            vals[-2] = vals[-1] - self.denom
        for i in range(1, len(vals)):
            if vals[i] <= vals[i - 1]:
                vals[i] = vals[i - 1] + self.denom
        counts = self._allocate_players(k)
        vals = self._reconcile_pool(vals, counts)
        p_sum = sum(counts)
        pool_sum = sum(v * c for v, c in zip(vals, counts))
        if p_sum != self.players:
            raise ValueError(f"Player count mismatch: {p_sum} != {self.players}")
        if abs(round(pool_sum * 100) / 100 - self.pool) > 0.005:
            raise ValueError(f"Pool mismatch: ${round(pool_sum, 2)} != ${self.pool}")
        for i in range(1, len(vals)):
            if vals[i] <= vals[i - 1]:
                raise ValueError(f"Value ordering violation at tier {i}")
        return k, vals, counts


# ─────────────────────────────────────────────────────────────────────────────
# METABASE CLIENT (lightweight, reusable)
# ─────────────────────────────────────────────────────────────────────────────

METABASE_BASE_URL = "https://baazigames.metabaseapp.com"
METABASE_DATABASE_ID = 133

def get_metabase_session():
    """Get or create Metabase session headers."""
    import requests
    api_key = st.session_state.get("metabase_api_key", "")
    if not api_key:
        return None
    return {"x-api-key": api_key}


def query_metabase(sql: str) -> pd.DataFrame:
    """Execute SQL against cp_prod via Metabase API."""
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
            json=payload,
            headers=headers,
            timeout=120,
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
    tournament_id,
    tournament_name,
    MIN(buy_in) AS buy_in,
    MIN(entry_fee) AS entry_fee,
    MIN(bounty_amount) AS bounty_amount,
    COUNT(DISTINCT user_id) AS player_count,
    SUM(buy_in) AS total_buyin_pool,
    SUM(bounty_won) AS total_bounties_paid,
    MIN(tournament_date) AS tournament_date,
    MIN(lobby_id) AS lobby_id
FROM lakehouse_gold.user_tournament_play_history
WHERE tournament_date BETWEEN DATE('{start}') AND DATE('{end}')
  AND lobby_id != 9
  AND bounty_amount > 0
GROUP BY tournament_id, tournament_name
HAVING COUNT(DISTINCT user_id) >= {min_players}
ORDER BY MIN(bounty_amount) DESC, total_buyin_pool DESC
LIMIT 50
"""

ALL_TOURNAMENT_QUERY = """
SELECT
    tournament_id,
    tournament_name,
    MIN(buy_in) AS buy_in,
    MIN(entry_fee) AS entry_fee,
    MIN(bounty_amount) AS bounty_amount,
    COUNT(DISTINCT user_id) AS player_count,
    SUM(buy_in) AS total_buyin_pool,
    MIN(tournament_date) AS tournament_date
FROM lakehouse_gold.user_tournament_play_history
WHERE tournament_date BETWEEN DATE('{start}') AND DATE('{end}')
  AND lobby_id != 9
  AND (buy_in + entry_fee) > 0
GROUP BY tournament_id, tournament_name
HAVING COUNT(DISTINCT user_id) >= {min_players}
ORDER BY total_buyin_pool DESC
LIMIT 50
"""


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT APP
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Mystery Bounty Simulator",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .stMetric > div { background: #1a1a2e; border-radius: 10px; padding: 12px 16px; }
    .stMetric label { color: #8892b0 !important; font-size: 0.85rem !important; }
    .stMetric [data-testid="stMetricValue"] { color: #64ffda !important; }
    div[data-testid="stSidebar"] { background: #0d1117; }
    .compare-badge { display: inline-block; padding: 2px 8px; border-radius: 12px;
        font-size: 0.75rem; font-weight: 600; margin-left: 6px; }
    .compare-a { background: #1f6feb33; color: #58a6ff; }
    .compare-b { background: #f0883e33; color: #f0883e; }
    /* Red Run Simulation button */
    button[kind="primary"], .stButton > button[kind="primary"],
    div.stButton > button[type="button"][kind="primary"] {
        background-color: #e63946 !important;
        border-color: #e63946 !important;
        color: white !important;
    }
    button[kind="primary"]:hover, .stButton > button[kind="primary"]:hover {
        background-color: #c1121f !important;
        border-color: #c1121f !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎯 Mystery Bounty Simulator")
st.caption("CoinPoker Internal — Simulate & fine-tune bounty prize distributions")

# ─── SIDEBAR: Engine Parameters ───
with st.sidebar:
    st.header("⚙️ Engine Parameters")

    st.subheader("Distribution Mode")
    mode = st.selectbox(
        "Curve shape",
        ["power_decay", "linear_decay", "gaussian_mid"],
        help="power_decay = pyramid (most common), linear_decay = staircase, gaussian_mid = bell curve",
    )
    strength = st.slider(
        "Curve strength",
        min_value=0.5, max_value=3.0, value=1.5, step=0.1,
        help="Higher = steeper curve. For power_decay: controls decay rate. For gaussian: controls spread.",
    )

    st.subheader("Multiplier Range")
    mult_start = st.slider(
        "Start multiplier (tier-to-tier at bottom)",
        min_value=1.1, max_value=4.0, value=2.0, step=0.1,
        help="Ratio between consecutive tiers at the low end",
    )
    mult_end = st.slider(
        "End multiplier (tier-to-tier at top)",
        min_value=1.1, max_value=4.0, value=1.5, step=0.1,
        help="Ratio between consecutive tiers at the high end",
    )
    if mult_end > mult_start:
        st.warning("End multiplier should be ≤ start multiplier. Swapping.")
        mult_start, mult_end = mult_end, mult_start

    st.subheader("Tier Settings")
    use_auto_k = st.checkbox("Auto-recommend K", value=True)
    manual_k = st.slider(
        "Manual K (number of tiers)",
        min_value=4, max_value=12, value=7,
        disabled=use_auto_k,
    )
    max_top = st.slider(
        "Max players in top 3 tiers",
        min_value=3, max_value=20, value=8,
        help="Scarcity cap: top 3 tiers combined cannot exceed this count",
    )
    denomination = st.number_input(
        "Denomination (USDT)",
        min_value=1, max_value=100, value=5,
        help="Values snap to this grid (e.g. $5, $10, $15...)",
    )

    st.divider()
    st.subheader("🔗 Metabase Connection")
    # Load API key from Streamlit secrets (for cloud deployment) or manual entry
    secrets_key = ""
    try:
        secrets_key = st.secrets.get("METABASE_API_KEY", "")
    except Exception:
        pass
    if secrets_key and "metabase_api_key" not in st.session_state:
        st.session_state["metabase_api_key"] = secrets_key
    api_key = st.text_input(
        "API Key",
        type="password",
        value=st.session_state.get("metabase_api_key", ""),
        help="Metabase API key for cp_prod access. Auto-loaded from secrets on Streamlit Cloud.",
    )
    if api_key:
        st.session_state["metabase_api_key"] = api_key
        st.success("Connected to cp_prod", icon="✅")
    else:
        st.warning("Enter API key or configure it in Streamlit Cloud secrets.")

# ─── Initialize session state ───
if "comparison_runs" not in st.session_state:
    st.session_state.comparison_runs = []


def run_simulation(pool, players, min_bounty, max_bounty, label=""):
    """Run the BountyArchitect engine and return results dict."""
    try:
        engine = BountyArchitect(
            pool=pool,
            players=players,
            min_bounty=min_bounty,
            max_bounty=max_bounty,
            mult_start=mult_start,
            mult_end=mult_end,
            mode=mode,
            strength=strength,
            max_top_tier_count=max_top,
            denomination=denomination,
        )
        k_val = None if use_auto_k else manual_k
        k, vals, counts = engine.build(k=k_val)

        # Build results
        tiers = []
        for i in range(k):
            tiers.append({
                "Tier": f"L{i+1}",
                "Bounty (USDT)": vals[i],
                "Players": counts[i],
                "Subtotal": round(vals[i] * counts[i], 2),
                "% of Pool": round(vals[i] * counts[i] / pool * 100, 1),
                "% of Players": round(counts[i] / players * 100, 1),
            })

        result = {
            "label": label,
            "k": k,
            "vals": vals,
            "counts": counts,
            "tiers": tiers,
            "pool": pool,
            "players": players,
            "min_bounty": min_bounty,
            "max_bounty": max_bounty,
            "warnings": engine.warnings,
            "params": {
                "mode": mode, "strength": strength,
                "mult_start": mult_start, "mult_end": mult_end,
                "denomination": denomination, "max_top": max_top,
            },
        }
        return result

    except Exception as e:
        st.error(f"Engine error: {e}")
        return None


def render_results(result):
    """Render simulation results with charts and metrics."""
    if result is None:
        return

    df = pd.DataFrame(result["tiers"])
    vals = result["vals"]
    counts = result["counts"]
    pool = result["pool"]
    players = result["players"]
    k = result["k"]

    # ── Key Metrics ──
    avg_bounty = pool / players
    median_idx = 0
    cumulative = 0
    for i, c in enumerate(counts):
        cumulative += c
        if cumulative >= players / 2:
            median_idx = i
            break
    median_bounty = vals[median_idx]
    max_ratio = vals[-1] / vals[0]
    top3_pct = sum(counts[-3:]) / players * 100 if k >= 3 else 0
    top3_pool_pct = sum(v * c for v, c in zip(vals[-3:], counts[-3:])) / pool * 100

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Tiers (K)", k)
    col2.metric("Avg Bounty", f"${avg_bounty:,.2f}")
    col3.metric("Median Bounty", f"${median_bounty:,.2f}")
    col4.metric("Max/Min Ratio", f"{max_ratio:,.1f}x")
    col5.metric("Top 3 Tiers", f"{sum(counts[-3:])} players ({top3_pct:.0f}%)")

    # ── Charts ──
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        # Bounty value bar chart
        fig_vals = go.Figure()
        colors = px.colors.sequential.Teal
        n_colors = len(colors)
        tier_colors = [colors[min(i * n_colors // k, n_colors - 1)] for i in range(k)]

        fig_vals.add_trace(go.Bar(
            x=[f"L{i+1}" for i in range(k)],
            y=vals,
            marker_color=tier_colors,
            text=[f"${v:,.2f}" for v in vals],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Bounty: $%{y:,.2f}<extra></extra>",
        ))
        fig_vals.update_layout(
            title="Bounty Values by Tier",
            xaxis_title="Tier",
            yaxis_title="Bounty Value (USDT)",
            template="plotly_dark",
            height=400,
            margin=dict(t=60, b=40),
        )
        st.plotly_chart(fig_vals, use_container_width=True)

    with chart_col2:
        # Player distribution (pyramid-style horizontal bar)
        fig_players = go.Figure()
        fig_players.add_trace(go.Bar(
            y=[f"L{i+1} (${vals[i]:,.0f})" for i in range(k)],
            x=counts,
            orientation="h",
            marker_color=tier_colors,
            text=[f"{c} ({c/players*100:.0f}%)" for c in counts],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Players: %{x}<extra></extra>",
        ))
        fig_players.update_layout(
            title="Player Allocation (Pyramid)",
            xaxis_title="Number of Players",
            yaxis_title="Tier",
            template="plotly_dark",
            height=400,
            margin=dict(t=60, b=40),
        )
        st.plotly_chart(fig_players, use_container_width=True)

    # ── Pool distribution pie ──
    subtotals = [round(v * c, 2) for v, c in zip(vals, counts)]
    fig_pie = go.Figure(data=[go.Pie(
        labels=[f"L{i+1}" for i in range(k)],
        values=subtotals,
        hole=0.4,
        marker=dict(colors=tier_colors),
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>$%{value:,.2f} (%{percent})<extra></extra>",
    )])
    fig_pie.update_layout(
        title="Pool Distribution by Tier",
        template="plotly_dark",
        height=350,
        margin=dict(t=60, b=20),
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # ── Distribution Table ──
    st.subheader("Distribution Table")
    st.dataframe(
        df.style.format({
            "Bounty (USDT)": "${:,.2f}",
            "Subtotal": "${:,.2f}",
            "% of Pool": "{:.1f}%",
            "% of Players": "{:.1f}%",
        }),
        use_container_width=True,
        hide_index=True,
    )

    total_row = {
        "Tier": "TOTAL",
        "Bounty (USDT)": "",
        "Players": players,
        "Subtotal": pool,
        "% of Pool": 100.0,
        "% of Players": 100.0,
    }
    st.caption(f"**Total: {players} players, ${pool:,.2f} USDT pool**")

    # Warnings
    if result["warnings"]:
        for w in result["warnings"]:
            st.warning(w)

    return df


# ─── TABS ───
tab_manual, tab_metabase, tab_compare = st.tabs([
    "📝 Manual Simulator",
    "🔌 Load from CoinPoker",
    "🔄 Compare Runs",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: Manual Simulator
# ═══════════════════════════════════════════════════════════════════════════════
with tab_manual:
    st.subheader("Enter Tournament Parameters")

    mcol1, mcol2 = st.columns(2)
    with mcol1:
        pool = st.number_input("Prize Pool (USDT)", min_value=100, max_value=1_000_000, value=20000, step=1000)
        min_bounty = st.number_input("Min Bounty (USDT)", min_value=1.0, max_value=10000.0, value=10.0, step=1.0)
    with mcol2:
        players = st.number_input("Players", min_value=4, max_value=10000, value=180, step=10)
        max_bounty = st.number_input("Max Bounty (USDT)", min_value=10.0, max_value=100000.0, value=2000.0, step=100.0)

    run_btn = st.button("🚀 Run Simulation", type="primary", use_container_width=True)

    if run_btn:
        result = run_simulation(pool, players, min_bounty, max_bounty, label="Manual")
        if result:
            st.session_state["last_manual_result"] = result
            render_results(result)

            # Export buttons
            exp_col1, exp_col2, exp_col3 = st.columns(3)
            df_export = pd.DataFrame(result["tiers"])
            with exp_col1:
                csv = df_export.to_csv(index=False)
                st.download_button("📥 Download CSV", csv, "bounty_distribution.csv", "text/csv")
            with exp_col2:
                json_str = json.dumps({
                    "k": result["k"], "vals": result["vals"], "counts": result["counts"],
                    "pool": result["pool"], "players": result["players"],
                    "params": result["params"], "warnings": result["warnings"],
                }, indent=2)
                st.download_button("📥 Download JSON", json_str, "bounty_config.json", "application/json")
            with exp_col3:
                if st.button("📌 Save to Comparison"):
                    st.session_state.comparison_runs.append(result)
                    st.success(f"Saved! ({len(st.session_state.comparison_runs)} runs stored)")

    elif "last_manual_result" in st.session_state:
        render_results(st.session_state["last_manual_result"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: Load from CoinPoker (Metabase)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_metabase:
    st.subheader("Load Real Tournament Data from cp_prod")
    st.caption("Pull bounty tournaments from CoinPoker's data lake and simulate mystery bounty distributions")

    if not st.session_state.get("metabase_api_key"):
        st.info("Enter your Metabase API key in the sidebar to connect to cp_prod.")
    else:
        lcol1, lcol2, lcol3, lcol4 = st.columns(4)
        with lcol1:
            start_date = st.date_input("Start date", value=date.today() - timedelta(days=7))
        with lcol2:
            end_date = st.date_input("End date", value=date.today())
        with lcol3:
            min_player_filter = st.number_input("Min players", min_value=4, max_value=500, value=20)
        with lcol4:
            bounty_only = st.checkbox("Bounty tournaments only", value=True,
                                       help="Filter to tournaments with bounty_amount > 0 (PKO, CoinHunter, etc.)")

        if st.button("🔍 Fetch Tournaments", type="primary"):
            with st.spinner("Querying cp_prod via Metabase..."):
                query_template = BOUNTY_TOURNAMENT_QUERY if bounty_only else ALL_TOURNAMENT_QUERY
                sql = query_template.format(
                    start=start_date.isoformat(),
                    end=end_date.isoformat(),
                    min_players=min_player_filter,
                )
                df_tournaments = query_metabase(sql)

            if not df_tournaments.empty:
                st.session_state["tournaments_df"] = df_tournaments
                st.session_state["tournaments_bounty_only"] = bounty_only
                st.success(f"Found {len(df_tournaments)} {'bounty ' if bounty_only else ''}tournaments")
            else:
                st.warning("No tournaments found for this date range. Try expanding the date range or lowering min players.")

        if "tournaments_df" in st.session_state and not st.session_state["tournaments_df"].empty:
            df_t = st.session_state["tournaments_df"]
            is_bounty_data = st.session_state.get("tournaments_bounty_only", True)

            # Show tournament picker
            def make_label(r):
                name = r['tournament_name']
                pcount = int(r['player_count'])
                pool_val = float(r['total_buyin_pool'])
                bounty = float(r.get('bounty_amount', 0))
                if bounty > 0:
                    return f"{name} — {pcount} players, ${pool_val:,.0f} pool, ${bounty:,.2f} bounty"
                return f"{name} — {pcount} players, ${pool_val:,.0f} pool"

            df_t["label"] = df_t.apply(make_label, axis=1)
            selected_label = st.selectbox("Select a tournament", df_t["label"].tolist())
            selected_row = df_t[df_t["label"] == selected_label].iloc[0]

            player_count = int(selected_row['player_count'])
            buy_in_val = float(selected_row['buy_in'])
            entry_fee_val = float(selected_row['entry_fee'])
            total_buyin = float(selected_row['total_buyin_pool'])
            bounty_amt = float(selected_row.get('bounty_amount', 0))
            bounties_paid = float(selected_row.get('total_bounties_paid', 0)) if 'total_bounties_paid' in selected_row.index else 0
            actual_bounty_pool = round(bounty_amt * player_count, 2)

            # Tournament info card
            info_cols = st.columns(5)
            info_cols[0].metric("Players", f"{player_count}")
            info_cols[1].metric("Buy-in", f"${buy_in_val:,.2f}")
            info_cols[2].metric("Entry Fee", f"${entry_fee_val:,.2f}")
            info_cols[3].metric("Starting Bounty", f"${bounty_amt:,.2f}")
            info_cols[4].metric("Bounty Pool", f"${actual_bounty_pool:,.2f}")

            if bounties_paid > 0:
                st.caption(f"**Actual bounties paid out:** ${bounties_paid:,.2f} USDT "
                           f"(PKO progressive — grows as players are eliminated)")

            st.divider()
            st.subheader("Mystery Bounty Simulation Parameters")
            st.caption("Use the actual bounty pool from this tournament, or adjust to simulate a mystery bounty variant.")

            bcol1, bcol2, bcol3 = st.columns(3)
            with bcol1:
                pool_mode = st.radio(
                    "Bounty pool source",
                    ["Use actual bounty pool", "Custom percentage", "Manual amount"],
                    help="Choose how to calculate the mystery bounty pool",
                )
            with bcol2:
                if pool_mode == "Use actual bounty pool":
                    sim_pool = actual_bounty_pool
                    st.info(f"Pool: ${sim_pool:,.2f} (bounty × players)")
                elif pool_mode == "Custom percentage":
                    bounty_pct = st.slider(
                        "% of total buy-in pool",
                        min_value=10, max_value=90, value=50, step=5,
                    )
                    sim_pool = round(total_buyin * bounty_pct / 100, 2)
                    st.info(f"Pool: ${sim_pool:,.2f} ({bounty_pct}% of ${total_buyin:,.2f})")
                else:
                    sim_pool = st.number_input(
                        "Manual bounty pool (USDT)",
                        min_value=10.0, max_value=500000.0,
                        value=float(actual_bounty_pool) if actual_bounty_pool > 0 else 1000.0,
                        step=100.0,
                    )

            with bcol3:
                mb_min = st.number_input(
                    "Min bounty (USDT)", min_value=0.10, max_value=10000.0,
                    value=max(0.50, round(bounty_amt * 0.1, 2)) if bounty_amt > 0 else 1.0,
                    step=0.50,
                    help="Lowest possible mystery bounty prize",
                )
                mb_max = st.number_input(
                    "Max bounty (USDT)", min_value=1.0, max_value=100000.0,
                    value=max(5.0, round(sim_pool * 0.10, 0)) if sim_pool > 0 else 100.0,
                    step=5.0,
                    help="Jackpot mystery bounty prize",
                )

            st.metric("Simulation Pool", f"${sim_pool:,.2f} USDT",
                       f"{player_count} players, ${sim_pool/player_count:,.2f} avg bounty")

            if st.button("🎯 Simulate Mystery Bounty", type="primary", use_container_width=True):
                result = run_simulation(
                    pool=sim_pool,
                    players=player_count,
                    min_bounty=mb_min,
                    max_bounty=mb_max,
                    label=f"{selected_row['tournament_name']}",
                )
                if result:
                    st.session_state["last_metabase_result"] = result
                    render_results(result)

                    exp_col1, exp_col2, exp_col3 = st.columns(3)
                    df_export = pd.DataFrame(result["tiers"])
                    with exp_col1:
                        csv = df_export.to_csv(index=False)
                        st.download_button("📥 Download CSV", csv, "bounty_distribution.csv",
                                           "text/csv", key="csv_mb")
                    with exp_col2:
                        json_str = json.dumps({
                            "tournament_id": int(selected_row['tournament_id']),
                            "tournament_name": selected_row['tournament_name'],
                            "k": result["k"], "vals": result["vals"], "counts": result["counts"],
                            "pool": result["pool"], "players": result["players"],
                            "original_bounty_amount": bounty_amt,
                            "params": result["params"], "warnings": result["warnings"],
                        }, indent=2)
                        st.download_button("📥 Download JSON", json_str, "bounty_config.json",
                                           "application/json", key="json_mb")
                    with exp_col3:
                        if st.button("📌 Save to Comparison", key="save_mb"):
                            st.session_state.comparison_runs.append(result)
                            st.success(f"Saved! ({len(st.session_state.comparison_runs)} runs stored)")

            elif "last_metabase_result" in st.session_state:
                render_results(st.session_state["last_metabase_result"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: Compare Runs
# ═══════════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.subheader("Compare Simulation Runs")
    runs = st.session_state.comparison_runs

    if len(runs) < 1:
        st.info("Run simulations and click '📌 Save to Comparison' to start comparing.")
    else:
        st.caption(f"{len(runs)} runs saved. Select two to compare side-by-side.")

        run_labels = [f"Run {i+1}: {r.get('label', 'Manual')} ({r['players']}p, ${r['pool']:,.0f})" for i, r in enumerate(runs)]

        ccol1, ccol2 = st.columns(2)
        with ccol1:
            idx_a = st.selectbox("Run A", range(len(runs)), format_func=lambda i: run_labels[i])
        with ccol2:
            idx_b = st.selectbox("Run B", range(len(runs)), index=min(1, len(runs)-1), format_func=lambda i: run_labels[i])

        run_a = runs[idx_a]
        run_b = runs[idx_b]

        # Side-by-side metrics
        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tiers", f"{run_a['k']} vs {run_b['k']}")
        avg_a = run_a['pool'] / run_a['players']
        avg_b = run_b['pool'] / run_b['players']
        m2.metric("Avg Bounty", f"${avg_a:,.2f} vs ${avg_b:,.2f}")
        ratio_a = run_a['vals'][-1] / run_a['vals'][0]
        ratio_b = run_b['vals'][-1] / run_b['vals'][0]
        m3.metric("Max/Min Ratio", f"{ratio_a:,.1f}x vs {ratio_b:,.1f}x")
        m4.metric("Top-3 Players",
                   f"{sum(run_a['counts'][-3:])} vs {sum(run_b['counts'][-3:])}")

        # Overlay chart
        fig_compare = go.Figure()
        fig_compare.add_trace(go.Bar(
            name=f"A: {run_a.get('label', 'Run A')}",
            x=[f"L{i+1}" for i in range(run_a['k'])],
            y=run_a['vals'],
            marker_color="#58a6ff",
            opacity=0.7,
        ))
        fig_compare.add_trace(go.Bar(
            name=f"B: {run_b.get('label', 'Run B')}",
            x=[f"L{i+1}" for i in range(run_b['k'])],
            y=run_b['vals'],
            marker_color="#f0883e",
            opacity=0.7,
        ))
        fig_compare.update_layout(
            title="Bounty Values — Side by Side",
            barmode="group",
            template="plotly_dark",
            height=400,
        )
        st.plotly_chart(fig_compare, use_container_width=True)

        # Parameter diff table
        st.subheader("Parameter Comparison")
        params_a = run_a.get("params", {})
        params_b = run_b.get("params", {})
        all_keys = sorted(set(list(params_a.keys()) + list(params_b.keys())))
        diff_rows = []
        for key in all_keys:
            va = params_a.get(key, "—")
            vb = params_b.get(key, "—")
            diff_rows.append({"Parameter": key, "Run A": str(va), "Run B": str(vb), "Changed": "✅" if va != vb else ""})
        st.dataframe(pd.DataFrame(diff_rows), use_container_width=True, hide_index=True)

        if st.button("🗑️ Clear all runs"):
            st.session_state.comparison_runs = []
            st.rerun()
