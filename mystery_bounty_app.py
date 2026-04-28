"""
Mystery Bounty Simulator — CoinPoker Internal Tool
====================================================
Streamlit app for simulating mystery bounty prize distributions.

Usage:
    pip install streamlit pandas plotly requests
    streamlit run mystery_bounty_app.py

Engine: BountyArchitect v18 — Power-Curve with Auto-Optimisation
Inputs: pool, players, min_bounty, max_bounty, start_multiplier
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
# SECTION 1: ENGINE — BountyArchitect v18 (Power-Curve)
# ═══════════════════════════════════════════════════════════════════════════════
#
# HOW IT WORKS:
#   1. User provides 5 inputs: pool, players, min, max, start_multiplier
#   2. Engine searches over (K, alpha, top_pattern, count_steepness, min_flex,
#      max_flex) to find the distribution that best satisfies all conditions:
#        C1: Hard rules — top tier = 1 player, values increasing, counts decreasing
#        C2: Pool match — total ≈ prize pool (< 3% drift, then fine-tuned)
#        C3: Count shape — linear at top (1,3,6 style), smooth decay to L1
#        C4: Value ratios — monotonically increasing from ~start_mult upward
#   3. Values use power-curve: v(t) = min × (max/min)^(t^α), α > 1 gives
#      increasing ratios naturally.
#   4. Counts use top-pattern + smooth exponential decay for lower tiers.
#
# PARAMETER RELAXATION (built into engine, not user-facing):
#   - Min bounty: ±10% of user input
#   - Max bounty: ±20% of user input
#   - Start multiplier: ±10% of user input
# ═══════════════════════════════════════════════════════════════════════════════

class BountyArchitect:
    """Mystery Bounty prize distribution engine v18 — Power-Curve.

    Generates a complete prize distribution with increasing value ratios,
    smooth count decay, and linear top-tier counts.

    Parameters:
        pool (float)       : Total bounty pool in USDT
        players (int)      : Number of players receiving bounties
        min_bounty (float) : Target smallest bounty (engine may flex ±10%)
        max_bounty (float) : Target largest bounty (engine may flex ±20%)
        mult_start (float) : Target starting multiplier (engine may flex ±10%)
    """

    DENOM = 10  # Values snap to nearest $10

    # Top-tier count patterns to search (reversed: [top, 2nd, 3rd])
    TOP_PATTERNS = [
        [1, 2, 3], [1, 2, 4], [1, 2, 5], [1, 2, 6],
        [1, 3, 5], [1, 3, 6], [1, 3, 7], [1, 3, 8],
        [1, 4, 7], [1, 4, 8],
    ]

    def __init__(self, pool, players, min_bounty, max_bounty, mult_start=2.0):
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
        self.mult_start = mult_start
        self.warnings = []

    # ── Value generation via power curve ──────────────────────────────────

    def _power_curve_values(self, k, mn, mx, alpha):
        """Generate bounty values using v(t) = mn × (mx/mn)^(t^α).

        α > 1 produces increasing tier-to-tier ratios (slow start,
        fast finish), which is the desired shape.

        Returns list of k values snapped to DENOM, or None if invalid.
        """
        d = self.DENOM
        vals = []
        for i in range(k):
            t = i / (k - 1) if k > 1 else 0
            v = mn * (mx / mn) ** (t ** alpha)
            v = round(v / d) * d
            vals.append(v)
        vals[0] = mn
        vals[-1] = mx

        # Fix non-strictly-increasing after rounding
        for i in range(1, k):
            if vals[i] <= vals[i - 1]:
                vals[i] = vals[i - 1] + d

        # Validate final value didn't drift too far from max
        if vals[-1] != mx:
            vals[-1] = mx
            if vals[-1] <= vals[-2]:
                return None

        return vals

    # ── Count generation ─────────────────────────────────────────────────

    def _build_counts(self, k, top_pattern, steep):
        """Build player counts: top tiers from pattern, rest via smooth decay.

        Args:
            k: number of tiers
            top_pattern: e.g. [1, 3, 6] for top-3 tier counts
            steep: multiplier for count growth going downward (1.3-2.7)

        Returns list of k counts, or None if invalid.
        """
        if k <= len(top_pattern):
            return None

        counts = [0] * k
        for j, c in enumerate(top_pattern):
            counts[k - 1 - j] = c

        # Build remaining tiers downward with smooth growth
        for i in range(k - len(top_pattern) - 1, -1, -1):
            above = counts[i + 1]
            counts[i] = max(above + 1, round(above * steep))

        # Adjust L1 to hit exact player count
        diff = self.players - sum(counts)
        counts[0] += diff

        # Validate
        if counts[0] <= 0 or counts[0] <= counts[1]:
            return None
        if any(counts[i] <= counts[i + 1] for i in range(k - 1)):
            return None

        return counts

    # ── Scoring ──────────────────────────────────────────────────────────

    # Preferred top patterns — lower index = more preferred
    PREFERRED_TOPS = [
        [1, 3, 6], [1, 3, 5], [1, 2, 5], [1, 2, 4], [1, 2, 3],
        [1, 3, 7], [1, 3, 8], [1, 4, 7], [1, 4, 8], [1, 2, 6],
    ]

    @staticmethod
    def _score(vals, counts, pool, mult_start_target):
        """Score a candidate distribution. Lower is better.

        Components:
          - Pool drift (most important)
          - Ratio monotonicity (must be increasing)
          - First ratio deviation from target start_mult
          - Count cliff factor (L1/L2 vs L2/L3 ratio)
          - Top-pattern linearity (prefer [1,3,6] style)
        """
        k = len(vals)
        total = sum(v * c for v, c in zip(vals, counts))
        drift = abs(pool - total) / pool * 100

        ratios = [vals[i + 1] / vals[i] for i in range(k - 1)]

        # Count ratio decreases (should be 0 for increasing ratios)
        n_dec = sum(1 for i in range(len(ratios) - 1)
                    if ratios[i + 1] < ratios[i] - 0.02)

        # First ratio deviation from target
        first_dev = abs(ratios[0] - mult_start_target)

        # Count cliff: L1/L2 vs L2/L3
        r12 = counts[0] / counts[1] if counts[1] > 0 else 999
        r23 = counts[1] / counts[2] if counts[2] > 0 else 999
        cliff = r12 / r23 if r23 > 0 else 999

        # Top-pattern preference: favor small, clean top-tier counts
        # [1,3,6]=sum 10 vs [1,4,7]=sum 12 vs [1,3,8]=sum 12
        # Also penalize L1 absorbing too many players (>50% of total)
        top3_sum = sum(counts[-3:])
        l1_share = counts[0] / sum(counts) if sum(counts) > 0 else 0

        score = (drift * 15
                 + n_dec * 80
                 + first_dev * 5
                 + max(0, cliff - 2.5) * 30
                 + max(0, top3_sum - 10) * 0.3
                 + max(0, l1_share - 0.45) * 10)

        return score, drift, ratios, n_dec

    # ── Fine-tune pool drift ─────────────────────────────────────────────

    def _fine_tune(self, vals, counts):
        """Nudge middle-tier values by DENOM steps to reduce pool drift.

        Preserves both value ordering AND ratio monotonicity.
        """
        d = self.DENOM
        k = len(vals)
        final = list(vals)
        drift = self.pool - sum(v * c for v, c in zip(final, counts))

        def _ratios_ok(v, idx):
            """Check that ratios around idx stay monotonically non-decreasing."""
            for j in range(max(0, idx - 1), min(len(v) - 1, idx + 1)):
                r_here = v[j + 1] / v[j]
                if j > 0:
                    r_prev = v[j] / v[j - 1]
                    if r_here < r_prev - 0.005:
                        return False
                if j + 2 < len(v):
                    r_next = v[j + 2] / v[j + 1]
                    if r_next < r_here - 0.005:
                        return False
            return True

        for _ in range(500):
            if abs(drift) < d:
                break
            step = d if drift > 0 else -d
            moved = False
            for i in range(1, k - 1):
                if abs(drift) < d:
                    break
                nv = final[i] + step
                if final[i - 1] < nv < final[i + 1]:
                    old_val = final[i]
                    final[i] = nv
                    if _ratios_ok(final, i):
                        new_drift = drift - step * counts[i]
                        if abs(new_drift) < abs(drift):
                            drift = new_drift
                            moved = True
                        else:
                            final[i] = old_val
                    else:
                        final[i] = old_val
            if not moved:
                break

        return final

    # ── Main build ───────────────────────────────────────────────────────

    def build(self):
        """Search for the optimal distribution and return results.

        Returns:
            tuple: (k, ideal_vals, final_vals, counts, meta)
              - k (int)             : Number of tiers
              - ideal_vals (list)   : Pre-fine-tune values
              - final_vals (list)   : Pool-corrected values
              - counts (list)       : Player count per tier
              - meta (dict)         : Engine metadata (actual min/max/mult used)
        """
        d = self.DENOM

        # Define search ranges with parameter relaxation
        min_lo = round(self.min_val * 0.90 / d) * d
        min_hi = round(self.min_val * 1.10 / d) * d
        min_range = list(range(max(d, min_lo), min_hi + d, d))
        if not min_range:
            min_range = [round(self.min_val / d) * d]

        max_lo = round(self.max_val * 0.80 / d) * d
        max_hi = round(self.max_val * 1.20 / d) * d
        max_step = max(d, round((max_hi - max_lo) / 20 / d) * d)
        max_range = list(range(max(d, max_lo), max_hi + max_step, max_step))
        if not max_range:
            max_range = [round(self.max_val / d) * d]

        # Adaptive drift tolerance: tighter pools get stricter limits
        # avg_bounty / max_bounty ratio indicates how top-heavy the pool is
        avg_bounty = self.pool / self.players
        top_heaviness = self.max_val / avg_bounty
        drift_tolerance = min(15.0, max(3.0, top_heaviness * 0.5))

        best_score = float('inf')
        best_result = None

        for K in range(5, 11):
            for top in self.TOP_PATTERNS:
                for steep_x10 in range(13, 28, 2):
                    steep = steep_x10 / 10.0
                    counts = self._build_counts(K, top, steep)
                    if counts is None:
                        continue

                    for mn in min_range:
                        for mx in max_range:
                            if mx <= mn:
                                continue

                            for alpha_x10 in range(11, 35, 1):
                                alpha = alpha_x10 / 10.0
                                vals = self._power_curve_values(K, mn, mx, alpha)
                                if vals is None:
                                    continue

                                # Quick checks before full scoring
                                ratios = [vals[i + 1] / vals[i]
                                          for i in range(K - 1)]

                                # First ratio must be within ±20% of target
                                if not (self.mult_start * 0.80 <= ratios[0]
                                        <= self.mult_start * 1.20):
                                    continue

                                # Ratios should be roughly increasing
                                n_dec = sum(
                                    1 for i in range(len(ratios) - 1)
                                    if ratios[i + 1] < ratios[i] - 0.02
                                )
                                if n_dec > 1:
                                    continue

                                score, drift, _, _ = self._score(
                                    vals, counts, self.pool, self.mult_start)

                                if drift > drift_tolerance:
                                    continue

                                if score < best_score:
                                    best_score = score
                                    best_result = {
                                        'k': K,
                                        'vals': vals[:],
                                        'counts': counts[:],
                                        'top': top,
                                        'alpha': alpha,
                                        'mn': mn,
                                        'mx': mx,
                                    }

        if best_result is None:
            # Fallback: use simple geometric with original params
            self.warnings.append(
                "Could not find optimal power-curve fit. "
                "Using fallback geometric distribution."
            )
            return self._fallback_build()

        k = best_result['k']
        ideal = best_result['vals']
        counts = best_result['counts']

        # Fine-tune pool drift
        final = self._fine_tune(ideal, counts)

        meta = {
            'actual_min': best_result['mn'],
            'actual_max': best_result['mx'],
            'alpha': best_result['alpha'],
            'top_pattern': best_result['top'],
            'score': best_score,
        }

        return k, ideal, final, counts, meta

    def _fallback_build(self):
        """Pool-aware geometric fallback if power-curve search fails.

        Iteratively adjusts max bounty downward (and min up if needed)
        until pool drift < 5%. Reports adjusted params as warnings.
        """
        d = self.DENOM
        best_fb = None
        avg = self.pool / self.players

        # Try scaling max down and min up
        # For small pools, min may need to rise significantly
        mn_mults = [1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0]
        for k in range(7, 4, -1):
            for mn_mult in mn_mults:
                mn = round(self.min_val * mn_mult / d) * d or d
                if mn >= avg:
                    continue
                for mx_mult in [x / 100.0 for x in range(100, 9, -5)]:
                    mx = round(self.max_val * mx_mult / d) * d
                    if mx <= mn * 3 or mx <= avg:
                        continue

                    ratio = (mx / mn) ** (1.0 / (k - 1))
                    vals = [mn]
                    for i in range(1, k):
                        vals.append(round(vals[-1] * ratio / d) * d)
                    vals[0] = mn
                    vals[-1] = mx
                    for i in range(1, k):
                        if vals[i] <= vals[i - 1]:
                            vals[i] = vals[i - 1] + d

                    for tp in self.TOP_PATTERNS[:6]:
                        if len(tp) > k:
                            continue
                        for steep in [1.3, 1.5, 2.0, 2.5]:
                            counts = self._build_counts(k, tp, steep)
                            if counts is None:
                                continue

                            total = sum(v * c for v, c in zip(vals, counts))
                            drift = abs(total - self.pool) / self.pool * 100

                            if best_fb is None or drift < best_fb[0]:
                                best_fb = (drift, k, vals[:], counts[:],
                                           mn, mx, tp)

        if best_fb and best_fb[0] < 20:
            _, k, vals, counts, mn, mx, tp = best_fb
            final = self._fine_tune(vals, counts)
            if mn != self.min_val or mx != self.max_val:
                self.warnings.append(
                    f"Inputs adjusted for feasibility: "
                    f"min ${self.min_val}→${mn}, "
                    f"max ${self.max_val:,}→${mx:,}"
                )
            meta = {
                'actual_min': mn,
                'actual_max': mx,
                'alpha': 1.0,
                'top_pattern': tp,
                'score': -1,
            }
            return k, vals, final, counts, meta

        # Last resort: scale everything from pool/players average
        k = 6
        mn = round(avg * 0.15 / d) * d or d
        mx = round(avg * 8 / d) * d
        ratio = (mx / mn) ** (1.0 / (k - 1))
        vals = [mn]
        for i in range(1, k):
            vals.append(round(vals[-1] * ratio / d) * d)
        vals[0] = mn
        vals[-1] = mx
        for i in range(1, k):
            if vals[i] <= vals[i - 1]:
                vals[i] = vals[i - 1] + d

        counts = [0] * k
        counts[-1] = 1
        counts[-2] = 2
        counts[-3] = max(3, min(5, self.players // 20))
        for i in range(k - 4, -1, -1):
            counts[i] = max(counts[i + 1] + 1,
                            round(counts[i + 1] * 1.5))
        diff = self.players - sum(counts)
        counts[0] += diff
        if counts[0] <= counts[1]:
            counts[0] = counts[1] + 1

        self.warnings.append(
            f"Original min/max infeasible for this pool. "
            f"Auto-adjusted to min=${mn}, max=${mx:,}"
        )
        final = self._fine_tune(vals, counts)
        meta = {
            'actual_min': mn,
            'actual_max': mx,
            'alpha': 1.0,
            'top_pattern': [1, 2, counts[-3]],
            'score': -1,
        }
        return k, vals, final, counts, meta


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
        "- Number of tiers (K)\n"
        "- End multiplier\n"
        "- Top-tier count pattern\n"
        "- Count curve steepness\n\n"
        "Parameter relaxation:\n"
        "- Min bounty: ±10%\n"
        "- Max bounty: ±20%\n"
        "- Start multiplier: ±10%"
    )


# ─── Session state ───
if "comparison_runs" not in st.session_state:
    st.session_state.comparison_runs = []


# ─── Simulation runner ───

def run_simulation(pool, players, min_bounty, max_bounty,
                   mult_start, label=""):
    """Run BountyArchitect v18 and return results dict."""
    try:
        engine = BountyArchitect(
            pool=pool, players=players,
            min_bounty=min_bounty, max_bounty=max_bounty,
            mult_start=mult_start,
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
            "mult_start": mult_start,
            "warnings": engine.warnings,
            "meta": meta,
            "params": {
                "mult_start": mult_start,
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

tab_manual, tab_metabase, tab_compare = st.tabs([
    "📝 Manual Simulator",
    "🔌 Load from CoinPoker",
    "🔄 Compare Runs",
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
        mult_start = st.number_input(
            "Start Multiplier",
            min_value=1.2, max_value=5.0, value=2.0, step=0.1,
            help="Target tier-to-tier ratio at L1→L2. Engine may flex ±10%.")
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
                mult_start, label="Manual")
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
                mb_mult = st.number_input(
                    "Start multiplier",
                    min_value=1.2, max_value=5.0, value=2.0, step=0.1)

            if st.button("🎯 Simulate", type="primary",
                          use_container_width=True):
                with st.spinner("Searching for optimal distribution..."):
                    result = run_simulation(
                        sim_pool, player_count, mb_min, mb_max,
                        mb_mult, label=row['tournament_name'])
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
