

from __future__ import annotations

from dataclasses import dataclass
import csv
import math
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


SUPPORTED_HORMONES: Tuple[str, ...] = ("ft4", "ft3", "tt3", "tsh")


_ALIASES: Dict[str, str] = {
    # canonical columns
    "rtf": "rtf",
    "ft4": "ft4",
    "ft3": "ft3",
    "tt3": "tt3",
    "tsh": "tsh",
    # common variants
    "freet4": "ft4",
    "freet3": "ft3",
    "totalt3": "tt3",
    "tshp": "tsh",
    # allow obvious uppercase variants after normalization
}


def _normalize_key(key: Any) -> str:
    # Drop whitespace and punctuation so headers like "Free T4" map cleanly.
    s = str(key).strip().lower()
    return "".join(ch for ch in s if ch.isalnum())


def _canonical_key(key: Any) -> Optional[str]:
    k = _normalize_key(key)
    return _ALIASES.get(k)


def _coerce_to_rows(sim_grid: Any) -> List[Dict[str, float]]:
    if sim_grid is None:
        raise ValueError("sim_grid is required")

    # pandas DataFrame-like: use to_dict for conversion without importing pandas
    if hasattr(sim_grid, "to_dict") and hasattr(sim_grid, "columns") and not isinstance(sim_grid, Mapping):
        try:
            records = sim_grid.to_dict(orient="records")  # type: ignore[attr-defined]
            return _coerce_to_rows(records)
        except TypeError:
            # older pandas: to_dict("records")
            records = sim_grid.to_dict("records")  # type: ignore[attr-defined]
            return _coerce_to_rows(records)

    if isinstance(sim_grid, Mapping):
        # dict of columns
        canonical_cols: Dict[str, Sequence[Any]] = {}
        for raw_key, col in sim_grid.items():
            ck = _canonical_key(raw_key)
            if ck is None:
                continue
            canonical_cols[ck] = col  # type: ignore[assignment]

        if "rtf" not in canonical_cols:
            raise ValueError("sim_grid must include an 'rtf' column")

        rtf_col = list(canonical_cols["rtf"])
        n = len(rtf_col)
        rows: List[Dict[str, float]] = []
        for i in range(n):
            row: Dict[str, float] = {"rtf": float(rtf_col[i])}
            for h in SUPPORTED_HORMONES:
                if h in canonical_cols and i < len(canonical_cols[h]):
                    v = canonical_cols[h][i]
                    if v is not None:
                        row[h] = float(v)
            rows.append(row)
        return rows

    if isinstance(sim_grid, (list, tuple)):
        rows = []
        for item in sim_grid:
            if not isinstance(item, Mapping):
                raise TypeError("sim_grid rows must be mappings (dict-like)")
            out: Dict[str, float] = {}
            for raw_key, raw_val in item.items():
                ck = _canonical_key(raw_key)
                if ck is None or raw_val is None:
                    continue
                out[ck] = float(raw_val)
            if "rtf" not in out:
                raise ValueError("each sim_grid row must include 'rtf'")
            rows.append(out)
        return rows

    raise TypeError("unsupported sim_grid type; expected mapping, list[dict], or DataFrame-like")


def _coerce_observed(observed_labs: Mapping[str, Any]) -> Dict[str, float]:
    if observed_labs is None:
        raise ValueError("observed_labs is required")
    obs: Dict[str, float] = {}
    for raw_key, raw_val in observed_labs.items():
        ck = _canonical_key(raw_key)
        if ck is None:
            continue
        if ck == "rtf":
            continue
        if raw_val is None:
            continue
        obs[ck] = float(raw_val)
    return obs


def _coerce_weights(weights: Optional[Mapping[str, Any]]) -> Dict[str, float]:
    if not weights:
        return {}
    out: Dict[str, float] = {}
    for raw_key, raw_val in weights.items():
        ck = _canonical_key(raw_key)
        if ck is None or ck == "rtf":
            continue
        if raw_val is None:
            continue
        out[ck] = float(raw_val)
    return out


def _coerce_sigmas(sigmas: Optional[Mapping[str, Any]]) -> Dict[str, float]:
    if not sigmas:
        return {}
    out: Dict[str, float] = {}
    for raw_key, raw_val in sigmas.items():
        ck = _canonical_key(raw_key)
        if ck is None or ck == "rtf":
            continue
        if raw_val is None:
            continue
        out[ck] = float(raw_val)
    return out


def _clamp_01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _chi2_ppf_df1(conf: float) -> float:
    if not (0.0 < conf < 1.0):
        raise ValueError("conf must be in (0, 1)")

    p = (conf + 1.0) / 2.0

    # Prefer stdlib exact inverse CDF if available (Python 3.8+)
    try:
        from statistics import NormalDist

        z = NormalDist().inv_cdf(p)  # type: ignore[attr-defined]
        return z * z
    except Exception:
        # Fallback: Acklam's approximation of inverse normal CDF.
        z = _norm_ppf_acklam(p)
        return z * z


def _norm_ppf_acklam(p: float) -> float:

    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0,1)")

    # Coefficients in rational approximations.
    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    )

    plow = 0.02425
    phigh = 1.0 - plow

    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        num = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        return num / den
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        num = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        return num / den

    q = p - 0.5
    r = q * q
    num = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
    den = (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    return num / den


def load_simulation_grid_csv(csv_path: str) -> List[Dict[str, float]]:

    rows: List[Dict[str, float]] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for raw_row in reader:
            row: Dict[str, float] = {}
            for k, v in raw_row.items():
                ck = _canonical_key(k)
                if ck is None or v is None or v == "":
                    continue
                row[ck] = float(v)
            if "rtf" not in row:
                raise ValueError("CSV row missing required 'rtf' column")
            rows.append(row)
    return rows


@dataclass(frozen=True)
class RtfIntervalResult:
    rtf_hat: float
    interval_low: float
    interval_high: float
    error_min: float
    error_threshold: float
    used_hormones: Tuple[str, ...]
    error_mode: str
    rule: str
    conf: Optional[float] = None
    delta: Optional[float] = None
    # Optional diagnostics
    rtf_grid: Optional[Tuple[float, ...]] = None
    error_grid: Optional[Tuple[float, ...]] = None
    accepted_mask: Optional[Tuple[bool, ...]] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "rtf_hat": self.rtf_hat,
            "rtf_interval": (self.interval_low, self.interval_high),
            "interval_low": self.interval_low,
            "interval_high": self.interval_high,
            "error_min": self.error_min,
            "error_threshold": self.error_threshold,
            "used_hormones": list(self.used_hormones),
            "error_mode": self.error_mode,
            "rule": self.rule,
            "conf": self.conf,
            "delta": self.delta,
            "rtf_grid": list(self.rtf_grid) if self.rtf_grid is not None else None,
            "error_grid": list(self.error_grid) if self.error_grid is not None else None,
            "accepted_mask": list(self.accepted_mask) if self.accepted_mask is not None else None,
        }


def estimate_rtf_interval(
    sim_grid: Any,
    observed_labs: Mapping[str, Any],
    *,
    weights: Optional[Mapping[str, Any]] = None,
    sigmas: Optional[Mapping[str, Any]] = None,
    conf: float = 0.95,
    delta: Optional[float] = None,
    rule: str = "auto",
    include_diagnostics: bool = False,
) -> RtfIntervalResult:

    rows = _coerce_to_rows(sim_grid)
    obs = _coerce_observed(observed_labs)
    if not obs:
        raise ValueError("observed_labs must include at least one of FT4/FT3/TT3/TSH")

    w = _coerce_weights(weights)
    s = _coerce_sigmas(sigmas)

    # Sort by rtf for connected-component extraction.
    rows = sorted(rows, key=lambda r: r["rtf"])

    # Determine which hormones we can use (both predicted and observed available).
    available_pred = {h for h in SUPPORTED_HORMONES if any(h in r for r in rows)}
    used = tuple(sorted(h for h in obs.keys() if h in available_pred))
    if not used:
        raise ValueError(
            "no overlap between observed labs and simulation grid; "
            "need at least one predicted hormone column among FT4/FT3/TT3/TSH"
        )

    # Decide error mode.
    can_normalize = all(h in s and s[h] > 0.0 for h in used)
    if can_normalize:
        error_mode = "normalized_wls"
    else:
        error_mode = "plain_wls"

    # Compute error grid.
    rtf_grid: List[float] = []
    err_grid: List[float] = []
    for r in rows:
        rtf_val = float(r["rtf"])
        rtf_grid.append(rtf_val)

        e = 0.0
        for h in used:
            if h not in r:
                e = float("inf")
                break
            resid = float(r[h]) - obs[h]
            weight = float(w.get(h, 1.0))
            if error_mode == "normalized_wls":
                sigma = float(s[h])
                e += weight * (resid / sigma) ** 2
            else:
                e += weight * resid**2
        err_grid.append(e)

    if not err_grid or all(math.isinf(e) for e in err_grid):
        raise ValueError("simulation grid produced no finite errors; check inputs")

    # Point estimate.
    idx_hat = min(range(len(err_grid)), key=lambda i: err_grid[i])
    e_min = float(err_grid[idx_hat])
    rtf_hat = float(rtf_grid[idx_hat])

    # Threshold rule.
    rule_in = rule.strip().lower()
    if rule_in not in ("auto", "profile", "delta"):
        raise ValueError("rule must be one of: auto, profile, delta")

    if rule_in == "auto":
        rule_used = "profile" if error_mode == "normalized_wls" else "delta"
    else:
        rule_used = rule_in

    if rule_used == "profile":
        if error_mode != "normalized_wls":
            raise ValueError("profile rule requires sigmas for all used hormones (normalized_wls)")
        thr = e_min + _chi2_ppf_df1(conf)
        delta_used = None
        conf_used = conf
    else:
        # Delta rule works with either error mode.
        if delta is None:
            # Heuristic default: 5% of a scale estimate, with a non-zero fallback.
            max_e = max(e for e in err_grid if not math.isinf(e))
            scale = e_min if e_min > 0.0 else (max_e - e_min)
            if scale <= 0.0:
                scale = 1.0
            delta_used = 0.05 * scale
        else:
            delta_used = float(delta)
            if delta_used < 0.0:
                raise ValueError("delta must be non-negative")
        thr = e_min + delta_used
        conf_used = None

    accepted = [e <= thr for e in err_grid]
    # rtf_hat must always be accepted under sane settings.
    accepted[idx_hat] = True

    # Extract connected component containing idx_hat.
    left = idx_hat
    while left - 1 >= 0 and accepted[left - 1]:
        left -= 1
    right = idx_hat
    while right + 1 < len(accepted) and accepted[right + 1]:
        right += 1

    # Interpolate boundaries against the threshold for cleaner endpoints.
    low = rtf_grid[left]
    if left > 0:
        i_rej = left - 1
        if not accepted[i_rej]:
            e_rej = err_grid[i_rej]
            e_acc = err_grid[left]
            x_rej = rtf_grid[i_rej]
            x_acc = rtf_grid[left]
            denom = (e_acc - e_rej)
            if denom != 0.0 and not (math.isinf(e_rej) or math.isinf(e_acc)):
                low = x_rej + (thr - e_rej) * (x_acc - x_rej) / denom

    high = rtf_grid[right]
    if right + 1 < len(rtf_grid):
        i_rej = right + 1
        if not accepted[i_rej]:
            e_acc = err_grid[right]
            e_rej = err_grid[i_rej]
            x_acc = rtf_grid[right]
            x_rej = rtf_grid[i_rej]
            denom = (e_rej - e_acc)
            if denom != 0.0 and not (math.isinf(e_rej) or math.isinf(e_acc)):
                high = x_acc + (thr - e_acc) * (x_rej - x_acc) / denom

    low = _clamp_01(float(low))
    high = _clamp_01(float(high))
    if low > high:
        low, high = high, low

    return RtfIntervalResult(
        rtf_hat=rtf_hat,
        interval_low=low,
        interval_high=high,
        error_min=e_min,
        error_threshold=float(thr),
        used_hormones=used,
        error_mode=error_mode,
        rule=rule_used,
        conf=conf_used,
        delta=delta_used,
        rtf_grid=tuple(rtf_grid) if include_diagnostics else None,
        error_grid=tuple(err_grid) if include_diagnostics else None,
        accepted_mask=tuple(accepted) if include_diagnostics else None,
    )

