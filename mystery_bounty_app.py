"""
Mystery Bounty Simulator — CoinPoker Internal Tool
====================================================
Streamlit app for simulating and fine-tuning mystery bounty prize distributions.
Usage:
    pip install streamlit pandas plotly requests
    streamlit run mystery_bounty_app.py
Metabase integration: loads real tournament data from cp_prod (DB 133, Athena)
to simulate mystery bounty distributions on actual CoinPoker tournaments.
TABLE OF CONTENTS (search for these section headers to jump around):
=====================================================================
  SECTION 1: ENGINE — BountyArchitect v15      (line ~35)
      - __init__              : Input validation & parameter storage
      - recommend_k           : Auto-calculate optimal number of tiers
      - _generate_weights     : Player distribution weights (strength-influenced)
      - _allocate_players     : Assign players to tiers with pyramid enforcement
      - _generate_ideal_values: Generate bounty values with dynamic strength curve
      - build                 : Main entry point — orchestrates the full pipeline
  SECTION 2: METABASE INTEGRATION               (line ~300)
      - query_metabase  : Execute SQL against cp_prod via Metabase API
      - BOUNTY_TOURNAMENT_QUERY : SQL for PKO/bounty tournaments
      - ALL_TOURNAMENT_QUERY    : SQL for all tournaments
  SECTION 3: STREAMLIT APP CONFIG & CSS          (line ~380)
      - Page config, theme, custom CSS styles
      - TO CHANGE BUTTON COLOR: search for "Red Run Simulation button"
      - TO CHANGE THEME COLORS: edit the .streamlit/config.toml file
  SECTION 4: SIDEBAR — Engine Parameters         (line ~415)
      - Distribution mode selector (power_decay, linear_decay, gaussian_mid)
      - Curve strength, multiplier range, tier settings
      - Metabase API key connection (auto-loads from Streamlit secrets)
  SECTION 5: SIMULATION & RENDERING FUNCTIONS    (line ~490)
      - run_simulation  : Wraps BountyArchitect and returns results dict
      - render_results  : Displays charts, metrics, and distribution table
  SECTION 6: TAB 1 — Manual Simulator            (line ~685)
      - User inputs: pool, players, min/max bounty
      - Run button, export (CSV/JSON), save to comparison
  SECTION 7: TAB 2 — Load from CoinPoker         (line ~730)
      - Fetches real tournament data from Metabase
      - Tournament picker, pool source modes, simulation
  SECTION 8: TAB 3 — Compare Runs                (line ~890)
      - Side-by-side comparison of saved simulation runs
      - Overlay charts and parameter diff table
CONFIGURATION:
==============
  - Metabase API Key : Set in Streamlit Cloud Secrets (METABASE_API_KEY)
  - Database         : cp_prod (DB ID 133, Athena engine)
  - Data source table: lakehouse_gold.user_tournament_play_history
  - lobby_id = 9     : Excluded (freerolls, not real-money tournaments)
ENGINE v15 NOTES — Dynamic Strength Control:
  - strength > 1.0 = 'Late' Spike/Drop (flat near start, sharp move at top)
  - strength < 1.0 = 'Early' Spike/Drop (sharp move at start, flat near end)
  - strength = 1.0 = Linear interpolation
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
# SECTION 1: ENGINE — BountyArchitect v15
# ═══════════════════════════════════════════════════════════════════════════════
# Final Unified Engine v15: Dynamic Strength Control for Multiplier Curves.
#
# HOW IT WORKS (high level):
#   1. recommend_k()            → Decide how many tiers (K) based on multiplier fit
#   2. _generate_ideal_values() → Generate bounty values with dynamic strength curve
#   3. _allocate_players()      → Distribute players across tiers (pyramid shape)
#   4. build()                  → Drift-correct values so total ≈ pool amount
#
# STRENGTH PARAMETER:
#   strength > 1.0 = 'Late' Spike/Drop (stays flat near start, sharp move at top)
#   strength < 1.0 = 'Early' Spike/Drop (sharp move at start, stays flat near end)
#   strength = 1.0 = Linear
#
# TO MODIFY THE DISTRIBUTION SHAPE:
#   - Change _generate_weights() to alter how players are spread across tiers
#   - Change _generate_ideal_values() to alter how bounty amounts grow tier-to-tier
#   - Change max_top_tier_count to limit how many players get top prizes
# ═══════════════════════════════════════════════════════════════════════════════

class BountyArchitect:
    """Mystery Bounty prize distribution engine v15.
    Final Unified Engine with Dynamic Strength Control for Multiplier Curves.
    Generates a complete prize distribution table for mystery bounty tournaments.

    Parameters:
        pool (float)            : Total bounty pool in USDT
        players (int)           : Number of players receiving bounties
        min_bounty (float)      : Smallest possible bounty prize
        max_bounty (float)      : Largest possible bounty prize (jackpot)
        mult_start (float)      : Tier-to-tier ratio at the low end (default 2.0)
        mult_end (float)        : Tier-to-tier ratio at the high end (default 3.0)
        mode (str)              : Distribution curve — "power_decay", "linear_decay", or "gaussian_mid"
        strength (float)        : Controls curve shape:
                                    > 1.0 = Late spike (flat start, sharp top)
                                    < 1.0 = Early spike (sharp start, flat top)
                                    = 1.0 = Linear
        max_top_tier_count (int): Max players allowed in top 3 tiers combined
        denomination (int)      : Values snap to this grid (e.g. 5 → $5, $10, $15...)

    Example:
        engine = BountyArchitect(pool=10000, players=100, min_bounty=10, max_bounty=1000)
        k, ideal, final, counts = engine.build()
    """

    def __init__(self, pool, players, min_bounty, max_bounty,
                 mult_start=2.0, mult_end=3.0, mode="power_decay",
                 strength=1.5, max_top_tier_count=8, denomination=10):
        if pool <= 0 or players <= 0:
            raise ValueError("Pool and players must be > 0")
        if players * min_bounty > pool:
            raise ValueError("Pool cannot cover min_bounty × players")
        if max_bounty <= min_bounty:
            raise ValueError("max_bounty must be > min_bounty")
        if max_top_tier_count < 3:
            raise ValueError("max_top_tier_count must be >= 3")
        if strength <= 0:
            raise ValueError("Strength must be > 0")
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
        """Auto-calculate the optimal number of tiers (K).
        Finds K where the geometric mean of the span fits strictly between
        mult_start and mult_end, allowing for smooth interpolation.
        Falls back to log-based estimate if no clean fit is found.
        Clamped to 5-12 range.

        Returns:
            int: Recommended number of tiers (K)
        """
        target_ratio = self.max_val / self.min_val
        r_min = min(self.mult_start, self.mult_end)
        r_max = max(self.mult_start, self.mult_end)
        for k in range(5, 13):
            steps = k - 1
            avg_needed = target_ratio ** (1.0 / steps)
            if r_min <= avg_needed <= r_max:
                return k
        return max(5, min(12, round(
            math.log(target_ratio) / math.log((self.mult_start + self.mult_end) / 2)
        )))

    def _generate_weights(self, k):
        """Generate player distribution weights for K tiers.
        Strength-influenced exponential weights — higher tiers get
        progressively fewer players. The base ratio is nudged by strength
        so a more aggressive strength also concentrates players more.

        Args:
            k (int): Number of tiers
        Returns:
            list[float]: Raw weights (NOT normalized — used by _allocate_players)
        """
        base_r = 1.45 + (self.strength * 0.15)
        return [base_r ** i for i in range(k - 2)]

    def _allocate_players(self, k):
        """Distribute players across K tiers using weighted allocation.
        Steps:
          1. Pin top tier = 1 player, second-top = 2 players
          2. Distribute remaining players proportional to weights
          3. Enforce strict monotonic pyramid (each tier ≥ tier above it)

        Args:
            k (int): Number of tiers
        Returns:
            list[int]: Player count per tier (index 0 = lowest, k-1 = highest)
        """
        weights = self._generate_weights(k)
        total_w = sum(weights)
        counts = [0] * k
        counts[-1] = 1
        counts[-2] = 2
        remaining = self.players - 3
        if remaining <= 0:
            return counts
        raw = [(w / total_w) * remaining for w in weights]
        counts[:k - 2] = [max(1, math.floor(v)) for v in raw]
        rem = remaining - sum(counts[:k - 2])
        fractions = [v - math.floor(v) for v in raw]
        for i in sorted(range(k - 2), key=lambda x: fractions[x], reverse=True)[:rem]:
            counts[i] += 1
        # Strict Pyramid Enforcement
        for _ in range(k * 20):
            changed = False
            for i in range(k - 1):
                if counts[i] < counts[i + 1]:
                    if counts[i + 1] > 1:
                        counts[i] += 1
                        counts[i + 1] -= 1
                        changed = True
            if not changed:
                break
        return counts

    def _generate_ideal_values(self, k):
        """Generate bounty values with dynamic strength curve.
        Uses a power-curve (t^p) to interpolate multipliers between
        mult_start and mult_end across K-1 steps. Pins both endpoints,
        enforces monotonicity, scales to exact min/max span, then
        compounds into final USDT values.

        Strength interpretation:
          p > 1.0 → flat start, sharp top (late spike)
          p < 1.0 → sharp start, flat top (early spike)
          p = 1.0 → linear

        Args:
            k (int): Number of tiers
        Returns:
            list[float]: Bounty values per tier (ascending, snapped to denomination)
        """
        steps = k - 1
        ratios = [0.0] * steps
        is_descending = (self.mult_start > self.mult_end)
        r_bound_start = self.mult_start
        r_bound_end = self.mult_end

        # 1. PIN ENDPOINTS
        ratios[0] = r_bound_start
        ratios[-1] = r_bound_end

        # 2. GENERATE DYNAMIC STRENGTH CURVE
        p = self.strength
        range_diff = r_bound_end - r_bound_start
        for i in range(1, steps - 1):
            t = i / (steps - 1)
            curve_val = range_diff * (t ** p)
            ratios[i] = r_bound_start + curve_val

        # 3. ENFORCE BOUNDS & MONOTONICITY
        safe_min = min(r_bound_start, r_bound_end)
        safe_max = max(r_bound_start, r_bound_end)
        ratios = [max(safe_min, min(safe_max, r)) for r in ratios]
        if is_descending:
            for i in range(steps - 1, 0, -1):
                if ratios[i] > ratios[i - 1]:
                    ratios[i] = ratios[i - 1]
        else:
            for i in range(1, steps):
                if ratios[i] < ratios[i - 1]:
                    ratios[i] = ratios[i - 1]

        # 4. SCALE TO EXACT SPAN
        target_prod = self.max_val / self.min_val
        current_prod = math.prod(ratios)
        if current_prod > 0:
            scale_factor = (target_prod / current_prod) ** (1.0 / steps)
            ratios = [max(safe_min, min(safe_max, r * scale_factor)) for r in ratios]

        # 5. RE-VALIDATE AFTER SCALING
        ratios = [max(safe_min, min(safe_max, r)) for r in ratios]
        if is_descending:
            for i in range(steps - 1, 0, -1):
                if ratios[i] > ratios[i - 1]:
                    ratios[i] = ratios[i - 1]
        else:
            for i in range(1, steps):
                if ratios[i] < ratios[i - 1]:
                    ratios[i] = ratios[i - 1]

        # 6. COMPOUND VALUES
        vals = [self.min_val]
        for r in ratios:
            vals.append(vals[-1] * r)
        vals = [round(v / self.denom) * self.denom for v in vals]
        vals[0] = self.min_val
        vals[-1] = self.max_val

        # Explicit L2 Pin
        vals[1] = round(self.min_val * self.mult_start / self.denom) * self.denom

        # Strictly increasing bounties
        for i in range(1, k):
            if vals[i] <= vals[i - 1]:
                vals[i] = vals[i - 1] + self.denom
        return vals

    def build(self, k=None):
        """Main entry point — build the complete bounty distribution.
        Orchestrates the full pipeline: K recommendation → player allocation →
        ideal value generation → drift correction.

        Drift correction: adjusts middle tiers by denomination steps to bring
        the total pool sum as close to the target as possible. Unlike v6's
        _reconcile_pool, this is a lighter single-pass correction — the
        ideal values serve as the display reference, and final values are
        the pool-corrected output.

        Args:
            k (int, optional): Override number of tiers. If None, auto-recommends.
        Returns:
            tuple: (k, ideal, final, counts)
              - k (int)           : Number of tiers used
              - ideal (list[float]): Ideal bounty values (pre-correction)
              - final (list[float]): Pool-corrected bounty values
              - counts (list[int]) : Player count per tier (index 0 = lowest)
        Raises:
            ValueError: If pool cannot be distributed within constraints
        """
        if k is None:
            k = self.recommend_k()
        counts = self._allocate_players(k)
        ideal = self._generate_ideal_values(k)
        drift = self.pool - sum(v * c for v, c in zip(ideal, counts))
        final = list(ideal)
        if abs(drift) > 0.5:
            step = self.denom if drift > 0 else -self.denom
            for i in range(2, k - 2):
                if abs(drift) < self.denom / 2:
                    break
                nv = final[i] + step
                if self.min_val < nv < self.max_val:
                    final[i] = nv
                    drift -= step * counts[i]
        return k, ideal, final, counts


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
    """Execute a SQL query against CoinPoker's cp_prod database via Metabase API.

    TO CHANGE THE DATABASE:
      - Change DB_ID below (133 = cp_prod on Athena)
      - Change METABASE_URL if using a different Metabase instance

    TO ADD NEW QUERIES:
      - Add a new SQL template string (like BOUNTY_TOURNAMENT_QUERY)
      - Call this function with the formatted SQL

    Args:
        sql (str): Raw SQL query string (Athena/Presto dialect)
    Returns:
        pd.DataFrame: Query results as a DataFrame (empty if error/no results)
    """
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


# --- SQL: Fetch bounty tournaments only (PKO, CoinHunter, etc.) ---
# Filter: bounty_amount > 0 ensures only tournaments with knockout bounties
# lobby_id != 9 excludes freerolls
# TO MODIFY: change the WHERE clause, add columns, or adjust the LIMIT
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
    .engine-badge {
        display: inline-block;
        background: linear-gradient(135deg, #1f6feb22, #64ffda22);
        border: 1px solid #64ffda44;
        border-radius: 6px;
        padding: 2px 10px;
        font-size: 0.75rem;
        color: #64ffda;
        font-weight: 600;
        margin-left: 8px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎯 Mystery Bounty Simulator")
st.caption("CoinPoker Internal — Simulate & fine-tune bounty prize distributions · Engine v15")

# ─── SIDEBAR: Engine Parameters ───
with st.sidebar:
    st.header("⚙️ Engine Parameters")
    st.caption("BountyArchitect v15 — Dynamic Strength Control")

    st.subheader("Distribution Mode")
    mode = st.selectbox(
        "Curve shape",
        ["power_decay", "linear_decay", "gaussian_mid"],
        help="power_decay = pyramid (most common), linear_decay = staircase, gaussian_mid = bell curve",
    )
    strength = st.slider(
        "Curve strength",
        min_value=0.1, max_value=5.0, value=1.5, step=0.1,
        help=(
            "Controls the shape of the multiplier curve:\n"
            "• > 1.0 = Late spike (flat near start, sharp jump at top)\n"
            "• < 1.0 = Early spike (sharp jump at start, flat at top)\n"
            "• = 1.0 = Linear interpolation"
        ),
    )

    # Strength visual hint
    if strength > 1.2:
        st.caption(f"📈 **Late spike** — multipliers accelerate toward top tiers")
    elif strength < 0.8:
        st.caption(f"📉 **Early spike** — multipliers jump fast, then flatten")
    else:
        st.caption(f"↔️ **Linear** — smooth multiplier progression")

    st.subheader("Multiplier Range")
    mult_start = st.slider(
        "Start multiplier (tier-to-tier at bottom)",
        min_value=1.1, max_value=5.0, value=2.0, step=0.1,
        help="Ratio between consecutive tiers at the low end (L1→L2)",
    )
    mult_end = st.slider(
        "End multiplier (tier-to-tier at top)",
        min_value=1.1, max_value=5.0, value=3.0, step=0.1,
        help="Ratio between consecutive tiers at the high end (top tiers)",
    )
    st.caption(
        f"Multipliers interpolate {mult_start:.1f}x → {mult_end:.1f}x "
        f"({'increasing' if mult_end > mult_start else 'decreasing' if mult_end < mult_start else 'flat'}) "
        f"via strength curve"
    )

    st.subheader("Tier Settings")
    use_auto_k = st.checkbox("Auto-recommend K", value=True)
    manual_k = st.slider(
        "Manual K (number of tiers)",
        min_value=5, max_value=12, value=7,
        disabled=use_auto_k,
    )
    max_top = st.slider(
        "Max players in top 3 tiers",
        min_value=3, max_value=20, value=8,
        help="Scarcity cap: top 3 tiers combined (top tier is always 1, second is always 2)",
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
    """Run the BountyArchitect v15 engine and return results dict.
    Returns a dict with both 'ideal' (pre-correction) and 'vals' (pool-corrected)
    tier values, plus counts, metrics, and engine params.
    """
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
        k, ideal, final, counts = engine.build(k=k_val)

        # Use final (pool-corrected) values for the distribution table
        tiers = []
        for i in range(k):
            tiers.append({
                "Tier": f"L{i+1}",
                "Ideal Bounty (USDT)": ideal[i],
                "Final Bounty (USDT)": final[i],
                "Players": counts[i],
                "Subtotal": round(final[i] * counts[i], 2),
                "% of Pool": round(final[i] * counts[i] / pool * 100, 1),
                "% of Players": round(counts[i] / players * 100, 1),
            })

        # Pool accuracy check
        pool_total = sum(v * c for v, c in zip(final, counts))
        pool_error = abs(pool_total - pool)

        result = {
            "label": label,
            "k": k,
            "ideal": ideal,
            "vals": final,       # pool-corrected — used for charts
            "counts": counts,
            "tiers": tiers,
            "pool": pool,
            "pool_total": round(pool_total, 2),
            "pool_error": round(pool_error, 2),
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
    ideal = result.get("ideal", vals)
    counts = result["counts"]
    pool = result["pool"]
    players = result["players"]
    k = result["k"]
    pool_error = result.get("pool_error", 0)

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

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Tiers (K)", k)
    col2.metric("Avg Bounty", f"${avg_bounty:,.2f}")
    col3.metric("Median Bounty", f"${median_bounty:,.2f}")
    col4.metric("Max/Min Ratio", f"{max_ratio:,.1f}x")
    col5.metric("Top 3 Tiers", f"{sum(counts[-3:])} players ({top3_pct:.0f}%)")

    # Pool accuracy info
    if pool_error > 0.01:
        st.info(
            f"ℹ️ **Pool drift:** ${pool_error:.2f} USDT — Final pool total is "
            f"${result['pool_total']:,.2f} vs target ${pool:,.2f}. "
            f"Middle tiers were adjusted by denomination steps to correct. "
            f"Ideal values (pre-correction) are shown in the table."
        )

    # ── Charts ──
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        # Bounty value bar chart — show both ideal and final if they differ
        fig_vals = go.Figure()
        colors = px.colors.sequential.Teal
        n_colors = len(colors)
        tier_colors = [colors[min(i * n_colors // k, n_colors - 1)] for i in range(k)]
        tier_labels = [f"L{i+1}" for i in range(k)]

        fig_vals.add_trace(go.Bar(
            name="Final (pool-corrected)",
            x=tier_labels,
            y=vals,
            marker_color=tier_colors,
            text=[f"${v:,.2f}" for v in vals],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Final: $%{y:,.2f}<extra></extra>",
        ))

        # Overlay ideal as a line if it differs meaningfully from final
        if any(abs(a - b) > 0.01 for a, b in zip(ideal, vals)):
            fig_vals.add_trace(go.Scatter(
                name="Ideal (pre-correction)",
                x=tier_labels,
                y=ideal,
                mode="lines+markers",
                line=dict(color="#f0883e", width=2, dash="dot"),
                marker=dict(size=6, color="#f0883e"),
                hovertemplate="<b>%{x}</b><br>Ideal: $%{y:,.2f}<extra></extra>",
            ))

        fig_vals.update_layout(
            title="Bounty Values by Tier",
            xaxis_title="Tier",
            yaxis_title="Bounty Value (USDT)",
            template="plotly_dark",
            height=400,
            margin=dict(t=60, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
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

    # ── Multiplier progression chart ──
    if k >= 3:
        multipliers = [vals[i] / vals[i - 1] for i in range(1, k)]
        ideal_multipliers = [ideal[i] / ideal[i - 1] for i in range(1, k)]

        fig_mult = go.Figure()
        fig_mult.add_trace(go.Scatter(
            x=[f"L{i}→L{i+1}" for i in range(1, k)],
            y=multipliers,
            mode="lines+markers",
            name="Final multipliers",
            line=dict(color="#64ffda", width=2),
            marker=dict(size=8),
            hovertemplate="<b>%{x}</b><br>Multiplier: %{y:.3f}x<extra></extra>",
        ))
        fig_mult.add_trace(go.Scatter(
            x=[f"L{i}→L{i+1}" for i in range(1, k)],
            y=ideal_multipliers,
            mode="lines+markers",
            name="Ideal multipliers",
            line=dict(color="#f0883e", width=2, dash="dot"),
            marker=dict(size=6),
            hovertemplate="<b>%{x}</b><br>Ideal mult: %{y:.3f}x<extra></extra>",
        ))
        fig_mult.add_hline(y=mult_start, line_dash="dash", line_color="#8892b0",
                           annotation_text=f"mult_start={mult_start:.1f}x", annotation_position="left")
        fig_mult.add_hline(y=mult_end, line_dash="dash", line_color="#8892b0",
                           annotation_text=f"mult_end={mult_end:.1f}x", annotation_position="left")
        fig_mult.update_layout(
            title=f"Tier-to-Tier Multiplier Progression (strength={strength:.1f})",
            xaxis_title="Tier Transition",
            yaxis_title="Multiplier",
            template="plotly_dark",
            height=300,
            margin=dict(t=60, b=40),
        )
        st.plotly_chart(fig_mult, use_container_width=True)

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
            "Ideal Bounty (USDT)": "${:,.2f}",
            "Final Bounty (USDT)": "${:,.2f}",
            "Subtotal": "${:,.2f}",
            "% of Pool": "{:.1f}%",
            "% of Players": "{:.1f}%",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        f"**Total: {players} players, ${pool:,.2f} USDT target pool, "
        f"${result['pool_total']:,.2f} actual pool** "
        f"{'(exact match ✅)' if pool_error < 0.01 else f'(drift: ${pool_error:.2f})'}"
    )

    # Warnings
    if result["warnings"]:
        for w in result["warnings"]:
            st.warning(w)

    return df


# ─── TABS ───
tab_manual, tab_metabase, tab_compare, tab_editor = st.tabs([
    "📝 Manual Simulator",
    "🔌 Load from CoinPoker",
    "🔄 Compare Runs",
    "✏️ Code Editor",
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
        players = st.number_input("Players", min_value=5, max_value=10000, value=180, step=10)
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
                    "k": result["k"],
                    "ideal": result["ideal"],
                    "vals": result["vals"],
                    "counts": result["counts"],
                    "pool": result["pool"],
                    "pool_total": result["pool_total"],
                    "pool_error": result["pool_error"],
                    "players": result["players"],
                    "params": result["params"],
                    "warnings": result["warnings"],
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
            min_player_filter = st.number_input("Min players", min_value=5, max_value=500, value=20)
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
                            "k": result["k"],
                            "ideal": result["ideal"],
                            "vals": result["vals"],
                            "counts": result["counts"],
                            "pool": result["pool"],
                            "pool_total": result["pool_total"],
                            "pool_error": result["pool_error"],
                            "players": result["players"],
                            "original_bounty_amount": bounty_amt,
                            "params": result["params"],
                            "warnings": result["warnings"],
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

        # Overlay chart — bounty values
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

        # Multiplier comparison chart
        if run_a['k'] >= 3 and run_b['k'] >= 3:
            mults_a = [run_a['vals'][i] / run_a['vals'][i-1] for i in range(1, run_a['k'])]
            mults_b = [run_b['vals'][i] / run_b['vals'][i-1] for i in range(1, run_b['k'])]

            fig_mults = go.Figure()
            fig_mults.add_trace(go.Scatter(
                name=f"A: {run_a.get('label', 'Run A')}",
                x=[f"L{i}→L{i+1}" for i in range(1, run_a['k'])],
                y=mults_a,
                mode="lines+markers",
                line=dict(color="#58a6ff", width=2),
            ))
            fig_mults.add_trace(go.Scatter(
                name=f"B: {run_b.get('label', 'Run B')}",
                x=[f"L{i}→L{i+1}" for i in range(1, run_b['k'])],
                y=mults_b,
                mode="lines+markers",
                line=dict(color="#f0883e", width=2),
            ))
            fig_mults.update_layout(
                title="Multiplier Progression Comparison",
                xaxis_title="Tier Transition",
                yaxis_title="Multiplier",
                template="plotly_dark",
                height=300,
            )
            st.plotly_chart(fig_mults, use_container_width=True)

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


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: Code Editor — Edit & Deploy this app directly
# ═══════════════════════════════════════════════════════════════════════════════
# Shows only the simulator-logic sections (engine, SQL, CSS) — not UI boilerplate.
# Edits are merged back into the full file and pushed to GitHub.
# ═══════════════════════════════════════════════════════════════════════════════

GITHUB_REPO = "sulemansafdar-rgb/mystery-bounty-simulator"
GITHUB_FILE = "mystery_bounty_app.py"
GITHUB_BRANCH = "main"

# --- Section definitions: (label, start_pattern, end_pattern) ---
# Each section is extracted from start_pattern up to (but not including) end_pattern.
EDITABLE_SECTIONS = [
    ("🎯 Tier Recommendation (recommend_k)",
     "    def recommend_k(self):", "    def _generate_weights(self, k):"),
    ("📊 Distribution Weights (_generate_weights)",
     "    def _generate_weights(self, k):", "    def _allocate_players(self, k):"),
    ("👥 Player Allocation (_allocate_players)",
     "    def _allocate_players(self, k):", "    def _generate_ideal_values(self, k):"),
    ("💡 Ideal Value Generation (_generate_ideal_values)",
     "    def _generate_ideal_values(self, k):", "    def build(self, k=None):"),
    ("🏗️ Build Pipeline (build)",
     "    def build(self, k=None):", "\n\n# ─────"),
    ("🔌 SQL: Bounty Tournament Query",
     "BOUNTY_TOURNAMENT_QUERY = \"\"\"", "ALL_TOURNAMENT_QUERY = \"\"\""),
    ("🔌 SQL: All Tournament Query",
     "ALL_TOURNAMENT_QUERY = \"\"\"", "\n\n# ─────"),
    ("🎨 CSS & Styling",
     "# Custom CSS\nst.markdown(\"\"\"", "\"\"\", unsafe_allow_html=True)"),
]


def extract_section(full_code: str, start: str, end: str) -> tuple:
    """Extract a section of code between start and end patterns.
    Returns (section_text, start_idx, end_idx) or (None, -1, -1) if not found.
    """
    s = full_code.find(start)
    if s == -1:
        return None, -1, -1
    e = full_code.find(end, s + len(start))
    if e == -1:
        e = len(full_code)
    return full_code[s:e], s, e


def replace_section(full_code: str, start_idx: int, end_idx: int, new_text: str) -> str:
    """Replace a section in the full code."""
    return full_code[:start_idx] + new_text + full_code[end_idx:]


def github_load_file(token: str) -> tuple:
    """Load a file from GitHub. Returns (content, sha) or (None, error_msg)."""
    import requests as _req
    import base64 as _b64
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github.v3+json"}
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_FILE}?ref={GITHUB_BRANCH}"
    try:
        resp = _req.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return _b64.b64decode(data["content"]).decode("utf-8"), data["sha"]
    except Exception as e:
        return None, str(e)


def github_push_file(token: str, content: str, sha: str, message: str = "") -> tuple:
    """Push updated file to GitHub. Returns (success, new_sha_or_error)."""
    import requests as _req
    import base64 as _b64
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json",
               "Accept": "application/vnd.github.v3+json"}
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_FILE}"
    payload = {
        "message": message or "Update simulator logic via Code Editor",
        "content": _b64.b64encode(content.encode("utf-8")).decode("ascii"),
        "sha": sha, "branch": GITHUB_BRANCH,
    }
    try:
        resp = _req.put(url, json=payload, headers=headers, timeout=30)
        if resp.status_code in (200, 201):
            return True, resp.json()["content"]["sha"]
        return False, f"GitHub error {resp.status_code}: {resp.json().get('message', '')}"
    except Exception as e:
        return False, str(e)


with tab_editor:
    st.subheader("✏️ Code Editor — Edit & Deploy")
    st.caption(
        "Edit the mystery bounty simulator logic below. "
        "Only the core logic sections are shown — UI boilerplate is hidden. "
        "Changes are merged into the full file and pushed to GitHub."
    )

    # --- GitHub Token (auto-load from secrets) ---
    gh_secrets_token = ""
    try:
        gh_secrets_token = st.secrets.get("GITHUB_PAT", "")
    except Exception:
        pass
    if gh_secrets_token and "github_pat" not in st.session_state:
        st.session_state["github_pat"] = gh_secrets_token

    gh_token = st.text_input(
        "GitHub Personal Access Token",
        type="password",
        value=st.session_state.get("github_pat", ""),
        help="Auto-loaded from Streamlit Secrets (GITHUB_PAT). Or paste manually.",
    )
    if gh_token:
        st.session_state["github_pat"] = gh_token

    if not gh_token:
        st.info(
            "🔑 **Setup required:** Enter a GitHub Personal Access Token above.\n\n"
            "**Quick setup:** Go to [github.com/settings/tokens](https://github.com/settings/tokens?type=beta) "
            "→ Generate new token → Select 'mystery-bounty-simulator' repo → Contents: Read & Write → Generate."
        )
    else:
        # --- Load / Push buttons ---
        ecol1, ecol2, ecol3 = st.columns([1, 1, 2])
        with ecol1:
            load_btn = st.button("📥 Load from GitHub", use_container_width=True)
        with ecol2:
            push_btn = st.button("🚀 Push & Deploy", type="primary", use_container_width=True)
        with ecol3:
            commit_msg = st.text_input(
                "Commit message", value="Update simulator logic via Code Editor",
                label_visibility="collapsed", placeholder="Commit message",
            )

        # --- Load ---
        if load_btn:
            with st.spinner("Loading from GitHub..."):
                content, sha_or_err = github_load_file(gh_token)
            if content is not None:
                st.session_state["editor_full_code"] = content
                st.session_state["editor_sha"] = sha_or_err
                st.session_state["editor_loaded"] = True
                # Extract all sections
                for i, (label, start_pat, end_pat) in enumerate(EDITABLE_SECTIONS):
                    section_text, _, _ = extract_section(content, start_pat, end_pat)
                    if section_text is not None:
                        st.session_state[f"section_{i}_original"] = section_text
                st.success(f"✅ Loaded! Showing {len(EDITABLE_SECTIONS)} editable logic sections.")
            else:
                st.error(f"Load failed: {sha_or_err}")

        # --- Show section editors ---
        if st.session_state.get("editor_loaded"):
            st.divider()
            st.caption("**Select a section to edit** — only simulator logic is shown, UI code is hidden.")

            # Section selector
            section_labels = [s[0] for s in EDITABLE_SECTIONS]
            selected_section = st.selectbox("Section", section_labels, key="section_picker")
            section_idx = section_labels.index(selected_section)

            label, start_pat, end_pat = EDITABLE_SECTIONS[section_idx]
            original_section = st.session_state.get(f"section_{section_idx}_original", "")

            if original_section:
                line_count = original_section.count("\n") + 1
                st.caption(f"**{label}** — {line_count} lines")

                # Editable text area for this section
                edited_section = st.text_area(
                    f"Edit: {label}",
                    value=original_section,
                    height=max(200, min(600, line_count * 18)),
                    key=f"editor_section_{section_idx}",
                    label_visibility="collapsed",
                )

                # Show change indicator
                if edited_section != original_section:
                    char_diff = len(edited_section) - len(original_section)
                    st.caption(f"📝 **Unsaved changes:** {'+' if char_diff >= 0 else ''}{char_diff} chars")
            else:
                st.warning(f"Section '{label}' not found in the code. Pattern may have changed.")
                edited_section = original_section

            # --- Push ---
            if push_btn:
                full_code = st.session_state.get("editor_full_code", "")
                sha = st.session_state.get("editor_sha")
                if not sha or not full_code:
                    st.error("Load the code from GitHub first.")
                else:
                    # Merge the edited section back into full code
                    _, s_idx, e_idx = extract_section(full_code, start_pat, end_pat)
                    if s_idx == -1:
                        st.error("Could not locate section in full code. Try reloading.")
                    elif edited_section == original_section:
                        st.warning("No changes to push.")
                    else:
                        merged = replace_section(full_code, s_idx, e_idx, edited_section)

                        # Validate Python syntax before pushing
                        import ast as _ast
                        try:
                            _ast.parse(merged)
                        except SyntaxError as syn_err:
                            st.error(f"⚠️ **Syntax error — not pushed.** Fix before deploying:\n\n`{syn_err}`")
                            merged = None

                        if merged:
                            with st.spinner("Pushing to GitHub..."):
                                success, result = github_push_file(gh_token, merged, sha, commit_msg)
                            if success:
                                st.session_state["editor_full_code"] = merged
                                st.session_state["editor_sha"] = result
                                st.session_state[f"section_{section_idx}_original"] = edited_section
                                st.success(
                                    "✅ **Pushed!** Streamlit Cloud will auto-deploy in ~1 minute. "
                                    "Refresh the app to see your changes."
                                )
                                st.balloons()
                            else:
                                st.error(f"Push failed: {result}")

            # --- Full code toggle (advanced) ---
            with st.expander("🔧 Advanced: View/Edit Full Code"):
                st.caption("⚠️ This shows the entire file including Streamlit UI code. Edit with care.")
                full_code_display = st.text_area(
                    "Full source code",
                    value=st.session_state.get("editor_full_code", ""),
                    height=400,
                    key="full_code_area",
                    label_visibility="collapsed",
                )
                if st.button("🚀 Push Full Code", key="push_full"):
                    sha = st.session_state.get("editor_sha")
                    if not sha:
                        st.error("Load first.")
                    else:
                        import ast as _ast
                        try:
                            _ast.parse(full_code_display)
                        except SyntaxError as syn_err:
                            st.error(f"Syntax error: {syn_err}")
                            full_code_display = None
                        if full_code_display:
                            with st.spinner("Pushing full code..."):
                                success, result = github_push_file(
                                    gh_token, full_code_display, sha, commit_msg
                                )
                            if success:
                                st.session_state["editor_full_code"] = full_code_display
                                st.session_state["editor_sha"] = result
                                st.success("✅ Full code pushed!")
                                st.balloons()
                            else:
                                st.error(f"Push failed: {result}")
        else:
            st.info("Click **📥 Load from GitHub** to start editing.")
