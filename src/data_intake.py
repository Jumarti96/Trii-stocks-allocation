"""
Data-intake helpers for pipeline step 1: load tickers, download Close+Volume in parallel batches,
and prune the universe by an activity filter (keep stocks that trade in >= min_active_fraction of
recent periods) plus the bad-data drop. Also resolves listing currencies and selects the modelling
universe (select_universe).

Currency handling, in one line: *ratios are currency-free, magnitudes are not.* Everything built on
pct_change() compares across markets safely; avg_dollar_volume does not, and must be converted
before it is ranked. See the "Currency resolution" section below for the measured consequences.

Lives in src/ (importable) so pipeline/01_download.py (digit-prefixed, not importable) stays a thin
orchestrator and the experiment/test suite can import these functions directly.
"""

import datetime
import glob as _glob
import re
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd


def load_tickers(csv_glob):
    """Load and clean tickers/ISINs from every CSV matching csv_glob.

    Reads utf-8-sig (strips BOM), coerces to str, strips whitespace, drops empty and 'nan',
    de-duplicates. Returns a list of any size.
    """
    out = set()
    for path in _glob.glob(csv_glob):
        col = pd.read_csv(path, header=None, encoding="utf-8-sig")[0]
        # .map(str), not .astype(str): under pandas 3.0's `str` dtype astype leaves
        # missing values as float NaN (2.x stringified them to 'nan'), so a literal
        # 'nan' token or an empty cell reaches .strip() as a float and raises.
        for raw in col.map(str).tolist():
            t = raw.strip()
            if t and t.lower() != "nan":
                out.add(t)
    return sorted(out)


def make_batches(tickers, batch_size):
    """Split a ticker list into consecutive batches of at most batch_size."""
    return [tickers[i:i + batch_size] for i in range(0, len(tickers), batch_size)]


def clean_batch(close_raw, volume_raw, period_freq, missing_frac=0.15):
    """Clean a batch's Close + Volume frames: period index, drop >missing_frac-missing Close
    tickers, ffill/bfill Close, align Volume to the kept names, string period-end index.

    Pure (no network). Returns (close, volume) over the same kept columns and string index.
    """
    close = close_raw.copy()
    volume = volume_raw.copy()
    close.index = close.index.to_period(freq=period_freq)
    volume.index = volume.index.to_period(freq=period_freq)
    close = close.groupby(level=0).first().sort_index()
    volume = volume.groupby(level=0).first().sort_index()

    keep = close.columns[close.isna().sum() < close.shape[0] * missing_frac]
    close = close[keep].ffill().bfill()
    volume = volume.reindex(columns=keep)

    close.index = close.index.astype("str").str.split("/").str[-1]
    volume.index = volume.index.astype("str").str.split("/").str[-1]
    return close, volume


def download_batch(batch, cfg):
    """Download one batch's Close+Volume from yfinance and clean it. Returns (close, volume) or None.

    Network. Extracts the Close and Volume sub-frames (multi-field columns for >1 ticker; single
    field set for 1 ticker), then clean_batch. One light retry on exception.
    """
    import yfinance as yf
    end = datetime.date.today()
    start = end - datetime.timedelta(days=cfg["days_of_data"])
    for attempt in (1, 2):
        try:
            raw = yf.download(batch, interval=cfg["interval"], start=start, end=end,
                              auto_adjust=True, threads=True, timeout=cfg["download_timeout"],
                              progress=False)
            if raw is None or raw.empty:
                return None
            if isinstance(raw.columns, pd.MultiIndex):
                close_raw, volume_raw = raw["Close"], raw["Volume"]
            else:  # single ticker -> flat columns
                close_raw = raw[["Close"]].rename(columns={"Close": batch[0]})
                volume_raw = raw[["Volume"]].rename(columns={"Volume": batch[0]})
            return clean_batch(close_raw, volume_raw, cfg["period_freq"])
        except Exception as e:  # noqa: BLE001 - batch-level resilience at scale
            if attempt == 1:
                print(f"  batch retry ({len(batch)} tickers): {e}")
            else:
                print(f"  batch failed ({len(batch)} tickers): {e}")
                return None


def download_all(tickers, cfg, download_fn=None):
    """Download all tickers in parallel batches and concat into aligned (close, volume) frames.

    download_fn(batch) -> (close, volume) | None is a DI seam (default: download_batch with cfg).
    Raises RuntimeError if every batch fails.
    """
    if download_fn is None:
        download_fn = lambda batch: download_batch(batch, cfg)  # noqa: E731
    batches = make_batches(tickers, cfg["batch_size"])
    closes, volumes = [], []
    with ThreadPoolExecutor(max_workers=cfg["download_workers"]) as ex:
        futures = {ex.submit(download_fn, b): b for b in batches if b}
        for fut in as_completed(futures):
            res = fut.result()
            if res is not None:
                c, v = res
                closes.append(c)
                volumes.append(v)
    if not closes:
        raise RuntimeError("No data downloaded across all batches.")
    close = pd.concat(closes, axis=1)
    volume = pd.concat(volumes, axis=1)
    # Guard against yfinance resolving two input identifiers to the same output symbol
    # (dual listings / ISIN aliases) -> duplicate columns crash the scalar per-ticker logic.
    duped = close.columns[close.columns.duplicated(keep=False)]
    if len(duped):
        print(f"  WARNING: dropped duplicate output columns (kept first): {sorted(set(duped))}")
        close = close.loc[:, ~close.columns.duplicated(keep="first")]
        volume = volume.loc[:, ~volume.columns.duplicated(keep="first")]
    return close, volume


# --- Currency resolution -------------------------------------------------------
#
# Returns are dimensionless ratios, so the module's original "currency-free" stance
# is correct for everything downstream of pct_change(). It breaks for *magnitudes*:
# avg_dollar_volume is Close * Volume in the listing currency, so comparing it
# across markets ranks by currency unit size rather than by liquidity. Measured on
# the 80-stock book, median ADV was CLP 9.8e9 / COP 6.4e9 / USD 5.8e9 / GBP 9.8e7,
# i.e. ~$10M / $1.6M / $5.8B / $124M -- and the top 30 by raw ADV contained zero
# GBP and zero CHF names. Anything that *ranks* by a magnitude must convert first.

CURRENCY_BY_SUFFIX = {
    "CL": "COP", "SN": "CLP", "SW": "CHF", "L": "GBP", "IL": "GBP",
    "DE": "EUR", "F": "EUR", "PA": "EUR", "AS": "EUR", "MC": "EUR", "MI": "EUR",
    "BR": "EUR", "LS": "EUR", "VI": "EUR", "HE": "EUR", "IR": "EUR", "AT": "EUR",
    "TO": "CAD", "V": "CAD", "NE": "CAD", "CN": "CAD",
    "T": "JPY", "AX": "AUD", "NZ": "NZD",
    "ST": "SEK", "CO": "DKK", "OL": "NOK", "IC": "ISK",
    "HK": "HKD", "SI": "SGD", "KS": "KRW", "KQ": "KRW",
    "SS": "CNY", "SZ": "CNY", "TW": "TWD", "TWO": "TWD",
    "NS": "INR", "BO": "INR", "JK": "IDR", "KL": "MYR", "BK": "THB",
    "SA": "BRL", "MX": "MXN", "BA": "ARS", "LM": "PEN",
    "TA": "ILS", "JO": "ZAR", "IS": "TRY", "WA": "PLN", "PR": "CZK", "BD": "HUF",
    "ME": "RUB", "CA": "EGP", "QA": "QAR", "SR": "SAR", "AD": "AED", "DU": "AED",
}

# ISIN country prefix -> currency. Used before download resolves ISINs to tickers.
# The prefix is the issuer's domicile, not the trading venue, so it is a fallback:
# KY/BM/VG issuers overwhelmingly list in the US, hence USD.
CURRENCY_BY_ISIN_COUNTRY = {
    "US": "USD", "CA": "CAD", "GB": "GBP", "JE": "GBP", "GG": "GBP", "IM": "GBP",
    "DE": "EUR", "FR": "EUR", "NL": "EUR", "IT": "EUR", "ES": "EUR", "BE": "EUR",
    "IE": "EUR", "FI": "EUR", "AT": "EUR", "PT": "EUR", "LU": "EUR", "GR": "EUR",
    "CH": "CHF", "JP": "JPY", "AU": "AUD", "NZ": "NZD",
    "SE": "SEK", "DK": "DKK", "NO": "NOK",
    "KY": "USD", "BM": "USD", "VG": "USD", "PA": "USD",
    "HK": "HKD", "SG": "SGD", "KR": "KRW", "CN": "CNY", "TW": "TWD", "IN": "INR",
    "BR": "BRL", "MX": "MXN", "CO": "COP", "CL": "CLP", "PE": "PEN", "AR": "ARS",
    "IL": "ILS", "ZA": "ZAR", "TR": "TRY", "PL": "PLN", "CZ": "CZK", "HU": "HUF",
}

_ISIN_RE = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$")


def infer_currency(symbol, overrides=None):
    """Return the listing currency for a ticker or ISIN, or None if unknown.

    Resolution order: explicit override -> yfinance exchange suffix -> ISIN country
    prefix -> bare symbol means a US listing (USD).

    Returns **None rather than a default** for anything unrecognised. Defaulting an
    unknown symbol to USD would let it be ranked by its raw local-currency magnitude,
    which is precisely the bug this module's currency handling exists to prevent; the
    caller is expected to exclude unknowns and say so.

    `overrides` handles genuinely ambiguous venues: the LSE lists USD-denominated
    lines (CSPX.L reports currency USD) alongside GBP ones, and no suffix rule can
    separate them. yfinance's `Ticker(sym).info['currency']` is authoritative when a
    caller is willing to pay for the lookup, and belongs in `overrides`.
    """
    if symbol is None:
        return None
    sym = str(symbol).strip()
    if not sym:
        return None
    if overrides and sym in overrides:
        return overrides[sym]
    if "." in sym:
        suffix = sym.rsplit(".", 1)[-1].upper()
        return CURRENCY_BY_SUFFIX.get(suffix)
    if _ISIN_RE.match(sym.upper()):
        return CURRENCY_BY_ISIN_COUNTRY.get(sym[:2].upper())
    return "USD"


# Quote currencies that are 1/100 of the currency FX is priced in. Yahoo returns
# these for a large share of LSE and Johannesburg listings, and treating them as the
# major unit overstates every amount 100x -- the single largest error available to a
# cross-market ranking. 'GBp' vs 'GBP' differ only by case, so that pair is matched
# case-sensitively; the others have no major-unit collision.
_MINOR_UNITS_EXACT = {"GBp": ("GBP", 0.01)}
_MINOR_UNITS_UPPER = {"ZAC": ("ZAR", 0.01), "ILA": ("ILS", 0.01),
                      "KWF": ("KWD", 0.001), "MGA": ("MGA", 1.0)}


def normalise_currency_code(raw):
    """Map a quote-currency code to (major_currency, unit_factor).

    unit_factor converts a quoted amount into the major unit, so
    `amount_major = amount_quoted * unit_factor`. Returns (None, 1.0) for a missing
    code so callers can distinguish "unknown" from "no scaling needed".
    """
    if not raw:
        return None, 1.0
    raw = str(raw).strip()
    if raw in _MINOR_UNITS_EXACT:
        return _MINOR_UNITS_EXACT[raw]
    up = raw.upper()
    if up in _MINOR_UNITS_UPPER:
        return _MINOR_UNITS_UPPER[up]
    return up, 1.0


def _yf_listing(identifier):
    """Listing metadata for one identifier via yfinance. None if unavailable."""
    import yfinance as yf
    try:
        return yf.Ticker(identifier).info
    except Exception:  # noqa: BLE001 - caller falls back to inference
        return None


RESOLVED_SOURCES = ("lookup",)          # rows a resume can safely keep


def resolve_listings(identifiers, fetch_fn=None, verbose=False, workers=1,
                     existing=None, retries=3, retry_wait=2.0, pause=0.0):
    """Authoritative listing metadata per identifier, falling back to inference.

    Returns a DataFrame indexed by the input identifier, with columns
    ['currency', 'unit_factor', 'symbol', 'name', 'source', 'sector', 'industry',
    'market_cap', 'exchange', 'quote_type'].

    `source` records HOW each row was obtained, and the distinction is load-bearing:
        'lookup'   - authoritative, from the provider
        'inferred' - the provider answered but named no currency; heuristic used
        'failed'   - the call did not come back; heuristic used, RETRY THIS ROW

    Separating the last two is what makes a rate-limited run recoverable. Collapsing
    them once let a 2,930-name catalogue finish "successfully" with 54% of its
    currencies guessed, and the guesses were worst exactly where the screen is most
    sensitive: NL0015000RT3 is NRP.JO quoted in Johannesburg cents, but its Dutch
    ISIN prefix inferred EUR, so it carried both the wrong rate and a missing 100x
    minor-unit factor -- enough to float it to rank 2 of a global liquidity ranking.

    existing: a previously written frame (i.e. 01_currency.csv). Rows whose source is
        'lookup' are reused verbatim and cost no network call; everything else --
        'failed', 'inferred', or absent -- is fetched again. Inference is a fallback,
        not an answer, so a later pass that can reach the network should replace it.
    retries/retry_wait: attempts per identifier and the base for exponential backoff.
        Rate limiting is transient; the default retry ladder rides out a short block
        rather than silently degrading to heuristics.
    pause: seconds between calls when single-threaded. Measured: three workers with no
        pause got blocked partway through 2,930 names, while one worker at 1.5s ran
        clean. Slower in theory, faster in practice than a run whose second half is
        guesswork.

    **Currency**: the exchange-suffix and ISIN-country tables are heuristics, measured
    at 87.5% over a 56-name stratified sample. The failures are not evenly spread:
    cross-listed ETFs are systematically wrong (CSPX.L and SGLD.L are USD-denominated
    despite .L; IUES.SW is USD despite .SW), Cayman and mainland-China ISINs are
    frequently HKD rather than USD/CNY, and Johannesburg quotes arrive in cents.
    Errors run 1.35x to 100x on exactly the magnitude the universe screen ranks by.
    On the 3,033-stock global universe the lookup resolved every name, and found 221
    quoted in minor units (195 GBp, 23 ZAc, 3 ILA) that inference would have
    overstated 100-fold.

    **Symbol**: yfinance labels its output columns with the *input* identifier, so an
    ISIN catalogue produces ISIN-labelled data all the way to the allocation report --
    'US67066G1040' rather than 'NVDA', which cannot be traded against. The symbol
    arrives in the same call as the currency, so capturing it is free.

    **Classification**: sector, industry, market_cap, exchange and quote_type ride
    along in the same response. They are what make a top-N liquidity screen auditable
    -- whether the survivors span the sectors, how much of the catalogue's market cap
    they cover, and whether ETFs (whose ADV dwarfs single stocks) are crowding real
    companies out of the ranking. market_cap is also the only data source for
    select_universe's min_market_cap gate.

    Costs one network round-trip per identifier (~0.6-1.6s), so it is issued across
    `workers` threads and cached: step 1 writes 01_currency.csv and step 2 only reads
    it. A single failed lookup falls back to inference rather than aborting the run --
    at 3,000 names, something will always fail. Note that concurrency and rate limits
    pull in opposite directions here: three workers over 2,930 names was enough to get
    blocked, and the recovery pass runs single-threaded on purpose.
    """
    if fetch_fn is None:
        fetch_fn = _yf_listing

    def one(ident):
        info, failed = {}, False
        for attempt in range(max(1, retries)):
            try:
                info = fetch_fn(ident) or {}
                failed = False
                break
            except Exception:  # noqa: BLE001 - a bad identifier must not sink the batch
                info, failed = {}, True
                if attempt + 1 < max(1, retries) and retry_wait:
                    # Exponential: a rate-limit block outlasts a flat retry, and
                    # hammering it is what extended the block in the first place.
                    time.sleep(retry_wait * (2 ** attempt))
        # A response that names neither a currency nor a symbol nor a name did not
        # actually answer. yfinance swallows an HTTP 401 internally and returns a
        # near-empty dict rather than raising, so exception-based detection misses it
        # completely: the 20-year download put 87 names on heuristic currencies, 10 of
        # them inside the top 300, while reporting 0 failures. Treat that as 'failed'
        # so it is retried and visible. A response that DOES identify the instrument
        # but omits the currency is a genuine gap, and inference is the right answer.
        identified = bool(info.get("symbol") or info.get("shortName")
                          or info.get("longName"))
        cur, factor = normalise_currency_code(info.get("currency"))
        source = "lookup"
        if cur is None:
            cur, factor = infer_currency(ident), 1.0
            source = "failed" if (failed or not identified) else "inferred"
        return {
            "currency": cur,
            "unit_factor": factor,
            "symbol": info.get("symbol") or ident,
            "name": info.get("shortName") or info.get("longName") or "",
            "source": source,
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            # Left as None when absent, never 0.0: yfinance reports no marketCap for
            # ETFs, and select_universe's min_market_cap gate keeps NaN on purpose so
            # they survive the screen. Zero-filling would delete every one of them.
            "market_cap": info.get("marketCap"),
            "exchange": info.get("exchange"),
            "quote_type": info.get("quoteType"),
        }

    ids = list(identifiers)
    rows, todo = {}, ids
    if existing is not None and len(existing):
        keep = existing.reindex([i for i in ids if i in existing.index])
        if "source" in keep.columns:
            keep = keep[keep["source"].isin(RESOLVED_SOURCES)]
            rows = {i: keep.loc[i].to_dict() for i in keep.index}
            todo = [i for i in ids if i not in rows]
            if verbose:
                print(f"  reusing {len(rows)} resolved rows, fetching {len(todo)}",
                      flush=True)

    if workers <= 1:
        for i, ident in enumerate(todo):
            rows[ident] = one(ident)
            if pause and i + 1 < len(todo):
                time.sleep(pause)
            if verbose and (i + 1) % 250 == 0:
                print(f"  listing {i + 1}/{len(todo)}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(one, ident): ident for ident in todo}
            for i, fut in enumerate(as_completed(futures)):
                rows[futures[fut]] = fut.result()
                if verbose and (i + 1) % 250 == 0:
                    print(f"  listing {i + 1}/{len(todo)}", flush=True)

    # Threads finish out of order; callers zip this against price columns.
    return pd.DataFrame.from_dict(rows, orient="index").reindex(ids)


def _default_fx_fetch(pair, index):
    """Download one Yahoo FX pair and align it to `index`. Returns None if unavailable."""
    import yfinance as yf
    try:
        raw = yf.download(pair, period="10y", interval="1wk",
                          auto_adjust=True, progress=False)
    except Exception:  # noqa: BLE001 - treated as "pair unavailable" by the caller
        return None
    if raw is None or raw.empty:
        return None
    close = raw["Close"]
    if hasattr(close, "columns"):
        close = close.iloc[:, 0]
    close.index = close.index.to_period("W").astype(str).str.split("/").str[-1]
    aligned = close.groupby(level=0).first().reindex(index).ffill().bfill()
    return None if aligned.isna().all() else aligned


def fetch_fx_rates(currencies, index, hub="USD", fetch_fn=None):
    """Per-period conversion rates into `hub`, as a DataFrame (index x currencies).

    rate[c] is "units of hub per 1 unit of c", so `amount_hub = amount_c * rate[c]`.

    Routed via USD rather than COP because the direct Colombian-peso crosses do not
    exist on Yahoo: CLPCOP=X returns 404 and CHFCOP=X is delisted, while USDCLP=X,
    USDCOP=X, GBPUSD=X and USDCHF=X all resolve. For each currency the direct pair
    (`{c}{hub}=X`) is tried first, then the reverse (`{hub}{c}=X`) inverted.

    Raises ValueError naming every currency it could not resolve. This is deliberate:
    filling an unavailable pair with 1.0 (or letting NaN become 0) leaves those names
    unconverted while the function reports success -- a silent degradation that
    reintroduces the raw-magnitude ranking bug for exactly the currencies whose data
    is hardest to get.
    """
    if fetch_fn is None:
        fetch_fn = _default_fx_fetch

    out, missing = {}, []
    for cur in dict.fromkeys(currencies):
        if cur is None:
            continue
        if cur == hub:
            out[cur] = pd.Series(1.0, index=index)
            continue
        direct = fetch_fn(f"{cur}{hub}=X", index)
        if direct is not None:
            out[cur] = direct
            continue
        reverse = fetch_fn(f"{hub}{cur}=X", index)
        if reverse is not None:
            out[cur] = 1.0 / reverse
            continue
        missing.append(cur)

    if missing:
        raise ValueError(
            f"No FX rate to {hub} for: {sorted(missing)}. Tried both "
            f"'<cur>{hub}=X' and '{hub}<cur>=X'. Refusing to continue -- an "
            f"unconverted currency would be ranked by its raw magnitude.")
    return pd.DataFrame(out, index=index)


def sanitise_fx(fx, max_move=2.0):
    """Repair single-period FX spikes that reverse immediately. Returns (fx, n_fixed).

    Provider FX series carry occasional bad ticks, and one of them corrupts every
    stock in that currency. Measured here: USDIDR=X reported 0.75 for the week of
    2017-05-07 against a true rate near 7.5e-5 -- exactly 10,000x out, a decimal
    slip -- which handed all six Indonesian names a +1,074,200% week followed by
    -100%. Those returns then propagate into the covariance matrix.

    A bad tick is identified as a move larger than `max_move` (as a ratio, so 2.0
    means tripling or worse) that REVERSES at the next period. Both halves are
    required: real currencies do crash, but they do not crash and fully recover in a
    week. The largest genuine weekly move in the measured 10-year panel was ZAR at
    42%, so the default sits far above anything real while still catching a 10,000x
    slip. Offending points are replaced by interpolation between their neighbours.

    A sustained devaluation is left alone at every step, since no step reverses.
    """
    out = fx.copy()
    n_fixed = 0
    for col in out.columns:
        s = out[col].astype(float)
        ratio = s / s.shift(1)
        for i in range(1, len(s) - 1):
            up, back = ratio.iloc[i], ratio.iloc[i + 1]
            if not (np.isfinite(up) and np.isfinite(back)):
                continue
            spiked = up > (1 + max_move) or up < 1 / (1 + max_move)
            reverted = (up > 1) != (back > 1) and abs(np.log(up * back)) < abs(np.log(up)) / 2
            if spiked and reverted:
                s.iloc[i] = (s.iloc[i - 1] + s.iloc[i + 1]) / 2.0
                ratio = s / s.shift(1)
                n_fixed += 1
        out[col] = s
    return out, n_fixed


def drop_bad_prices(close):
    """Replace non-positive prices with the last good one. Returns (close, n_fixed).

    An adjusted close of 0 or below is never legitimate -- it is a hole in the feed.
    Carried into pct_change it produces -1.0 and then +inf, and the infinity is what
    actually halts a run: LedoitWolf refuses to fit a matrix containing one, several
    hours into a backtest. Measured: one such tick on the 2,923-name catalogue
    (GB00B9XQT119, a single 0.0 between two prices near 141).

    Forward-fill rather than drop the name: one bad print is not a reason to discard
    ten years of history. A leading bad value is back-filled, since there is nothing
    earlier to carry forward.
    """
    bad = close <= 0
    n_fixed = int(bad.values.sum())
    if not n_fixed:
        return close, 0
    return close.mask(bad).ffill().bfill(), n_fixed


def drop_implausible_names(rets, max_return=5.0):
    """Remove stocks whose return series contains an impossible move. (rets, dropped).

    Unlike sanitise_fx and drop_bad_prices, this does not repair -- it excludes. The
    defect it catches is a quote-unit switch inside the provider's own price series:
    III.L, MEGP.L, OTV2.L, SLM.JO and PPH.JO each jump by almost exactly 100x
    partway through their history, a GBp/GBP or ZAc/ZAR flip. unit_factor is a
    single value per name and cannot express a switch that happens mid-series, so
    there is nothing to correct with; the honest move is to drop the name and say so.

    Excluding matters even when such a name looks harmless. A 100x price jump also
    inflates Close * Volume by 100x for the rest of the history, which is what
    select_universe ranks by -- so a corrupted line is actively pushed TOWARD the
    modelled universe, and once inside it dominates both the covariance matrix and
    any cross-sectional ranking the model learns.

    max_return defaults to 5.0 (+400% in one period), far above real extremes: a
    biotech can triple on trial results, but it does not gain 10,000%. Measured on
    the 2,923-name catalogue: 26 names exceed it, 0.9% of the universe.
    """
    worst = rets.abs().max()
    dropped = sorted(worst.index[worst > max_return])
    if not dropped:
        return rets, []
    return rets.drop(columns=dropped), dropped


UNKNOWN_CURRENCY_POLICIES = ("exclude", "assume_target")


def convert_currency(amounts, cur_map, fx, target=None, when=-1, unit_factors=None,
                     unknown="exclude"):
    """Convert per-ticker local-currency amounts into `target`.

    amounts: Series indexed by ticker.  cur_map: {ticker: currency}.
    fx: DataFrame from fetch_fx_rates, whose values are "units of the hub per 1 unit
        of the column currency".
    target: currency to convert into; None means the fx hub itself. The hub cancels,
        so the rate is simply fx[local] / fx[target].
    when: row of `fx` to use (default: last).
    unit_factors: optional {ticker: factor} for minor-unit quotes (pence, cents) --
        see normalise_currency_code. Absent, every quote is assumed to be in the major
        unit, which overstates pence-quoted names 100x.
    unknown: what to do with a ticker whose currency is unresolved or has no FX rate.
        'exclude' (default) drops it; 'assume_target' takes the number at face value
        as though already denominated in `target`.

    'exclude' is the default because guessing is how a JPY-quoted name gets ranked
    ~150x too high -- the precise failure this module's currency handling exists to
    prevent. 'assume_target' is offered so a single unresolvable ticker cannot abort a
    run, but it trades a visible gap for an invisible error, so callers should report
    how many names it touched.
    """
    if unknown not in UNKNOWN_CURRENCY_POLICIES:
        raise ValueError(
            f"unknown_currency policy must be one of {UNKNOWN_CURRENCY_POLICIES}, "
            f"got {unknown!r}")

    rates = fx.iloc[when]
    denom = 1.0 if target is None else float(rates[target])

    scale = {}
    for t in amounts.index:
        cur = cur_map.get(t)
        factor = 1.0 if unit_factors is None else unit_factors.get(t, 1.0)
        if cur in rates.index:
            scale[t] = float(rates[cur]) * factor / denom
        elif unknown == "assume_target":
            scale[t] = factor
        else:
            scale[t] = None
    return (amounts * pd.Series(scale, dtype="float64")).dropna()


def convert_panel(prices, cur_map, fx, target=None, unit_factors=None,
                  unknown="exclude"):
    """Convert a whole price panel into `target`, each period at its own FX rate.

    prices: DataFrame (period x ticker) of local-currency prices. Arguments otherwise
    match convert_currency, of which this is the per-period generalisation: that one
    converts a snapshot at a single `when`, this one converts every row.

    The difference matters for returns. pct_change() on native prices treats a flat
    COP-quoted stock as a flat holding, when a USD investor holding it through a 20%
    peso depreciation lost 20%. Converting first puts the FX move inside the return,
    which is what makes a multi-currency universe summable at all -- and what makes a
    comparison against a USD benchmark such as the S&P 500 mean anything.

    Tickers whose currency is unresolved or has no FX column are DROPPED under the
    default 'exclude' policy -- whole columns, not scattered NaNs, so a caller that
    zips the result against another frame cannot silently misalign.

    Raises ValueError if `fx` does not cover every period in `prices`. Reindexing with
    a gap would convert those rows at a neighbouring week's rate and manufacture a
    return that never happened, so the gap is reported rather than filled.
    """
    if unknown not in UNKNOWN_CURRENCY_POLICIES:
        raise ValueError(
            f"unknown_currency policy must be one of {UNKNOWN_CURRENCY_POLICIES}, "
            f"got {unknown!r}")

    missing_periods = [p for p in prices.index if p not in fx.index]
    if missing_periods:
        raise ValueError(
            f"fx is missing {len(missing_periods)} of the panel's periods, e.g. "
            f"{missing_periods[:5]}. Refusing to convert: filling the gap would "
            f"price those rows at another period's rate.")

    rates = fx.loc[prices.index]
    denom = 1.0 if target is None else rates[target]

    scales, dropped = {}, []
    for t in prices.columns:
        cur = cur_map.get(t)
        factor = 1.0 if unit_factors is None else unit_factors.get(t, 1.0)
        if cur in rates.columns:
            scales[t] = rates[cur] * factor / denom
        elif unknown == "assume_target":
            scales[t] = pd.Series(factor, index=prices.index)
        else:
            dropped.append(t)

    kept = [t for t in prices.columns if t not in dropped]
    if not kept:
        return prices.iloc[:, :0].copy()
    scale = pd.DataFrame(scales)[kept]
    return prices[kept] * scale


def avg_dollar_volume(close, volume, window, fx=None, cur_map=None, as_of=None,
                      unit_factors=None, unknown="exclude"):
    """Mean of (Close * Volume) over `window` periods, per ticker. NaN treated as 0.

    fx/cur_map: when both are supplied, the result is converted into the hub currency
    of `fx` (see to_hub_currency). Omitted, the value stays in each ticker's listing
    currency and is therefore only comparable within a single market.

    as_of: label in `close.index` marking the last period to include. The default
    (None) uses the tail, which is correct for production -- today is when the
    allocation is made. Backtests must pass the rebalance date instead: selecting a
    2019 universe by 2026 liquidity picks names we now know grew into mega-caps
    (NVDA's 2019 ADV was a fraction of today's), inflating every result. This bounds
    look-ahead among surviving names only; it cannot recover delisted ones.
    """
    dv = (close * volume).fillna(0.0)
    if as_of is not None:
        pos = list(dv.index).index(as_of) + 1
        dv = dv.iloc[:pos]
    adv = dv.iloc[-window:].mean(axis=0)
    if fx is not None and cur_map is not None:
        adv = convert_currency(adv, cur_map, fx, unit_factors=unit_factors,
                               unknown=unknown)
    return adv


def active_fraction(volume, window):
    """Per ticker, the fraction of the last `window` periods with real (Volume > 0) trading.

    NaN volume counts as not-traded (NaN > 0 is False). Returns a Series in [0, 1] -- a currency-free,
    unitless activity measure (no grouping, no magnitude).
    """
    return (volume.iloc[-window:] > 0).mean(axis=0)


def activity_filter(close, volume, window=None, min_active_fraction=0.85):
    """Keep stocks that trade in at least `min_active_fraction` of the last `window` periods.

    window defaults to the last 10% of the time-series (min 10 periods). Pass an explicit
    integer to override (useful in tests with small fixtures).
    Returns a per-ticker DataFrame [avg_dollar_volume (informational), active_fraction, kept].
    """
    if window is None:
        window = max(10, len(close) // 10)
    adv = avg_dollar_volume(close, volume, window)
    af = active_fraction(volume, window)
    detail = pd.DataFrame({
        "avg_dollar_volume": adv,
        "active_fraction": af,
        "kept": af >= min_active_fraction,
    })
    return detail


def activity_health(detail):
    """Summarise the activity filter: counts plus the share of stocks that never trade.

    zero_volume_fraction (active_fraction == 0) is the data-source alarm: if it is high, Volume is
    probably missing from the feed rather than the stocks being genuinely inactive.
    """
    n_total = len(detail)
    n_kept = int(detail["kept"].sum())
    zero_volume_fraction = float((detail["active_fraction"] == 0).mean()) if n_total else 0.0
    return {
        "n_total": n_total,
        "n_kept": n_kept,
        "n_excluded": n_total - n_kept,
        "zero_volume_fraction": zero_volume_fraction,
    }


def _stratified_pick(ranked, topn, strata):
    """Take strata[i] names from band i of `ranked` (already sorted best-first).

    `ranked` is split into len(strata) equal-sized bands. A band that cannot fill its
    quota passes the shortfall to the next band, so the total stays at topn whenever
    the universe is large enough -- silently returning fewer names would look like a
    smaller universe rather than an exhausted stratum.
    """
    n_bands = len(strata)
    edges = [round(i * len(ranked) / n_bands) for i in range(n_bands + 1)]
    picked, carry = [], 0
    for i, quota in enumerate(strata):
        band = [t for t in ranked[edges[i]:edges[i + 1]] if t not in picked]
        want = quota + carry
        picked.extend(band[:want])
        carry = want - len(band[:want])
    if carry:                       # trailing bands exhausted: backfill from the top
        picked.extend([t for t in ranked if t not in picked][:carry])
    return picked[:topn]


def select_universe(close, volume, topn, *, strata=None, window=None,
                    price_floor=0.0, min_active_fraction=0.0,
                    fx=None, cur_map=None, as_of=None, unit_factors=None,
                    unknown="exclude", market_cap=None, min_market_cap=None):
    """Choose which tickers to model. Returns a list of column names.

    Two phases: hard eligibility gates, then a liquidity ranking with an optional
    stratified budget.

    **Every criterion here is return-neutral by construction.** Momentum, past
    return, growth and any other "which of these will go up" signal are deliberately
    excluded: screening on them hands the model a universe of pre-selected winners,
    which is the optimistic bias pipeline/02_predict.py warns about, and it destroys
    the ability to tell whether the model has skill. Picking winners is the job of
    the forecast plus allocation_top_n, downstream of this function. Liquidity and
    size are a *style* tilt (toward large caps), not a return forecast.

    topn=None disables the screen entirely -- every criterion included -- so the
    pipeline reproduces its pre-screen universe exactly.

    strata: budget across equal-sized bands of the ADV ranking, e.g. [200, 60, 40]
        takes 200 names from the most liquid third, 60 from the middle third and 40
        from the least liquid third. None means a pure top-N, which at a 3.9k
        catalogue returns mega-caps only and no mid-cap exposure.
    fx/cur_map: passed to avg_dollar_volume so the ranking compares real liquidity
        rather than currency unit size. Names of unknown currency are dropped.
    market_cap: optional Series. Missing values are KEPT, not treated as zero --
        yfinance reports marketCap=None for ETFs, and zero-filling would delete
        every ETF from the universe.
    """
    if topn is None:
        return list(close.columns)

    if window is None:
        window = max(10, len(close) // 10)

    eligible = list(close.columns)

    if min_active_fraction > 0:
        af = active_fraction(volume[eligible], window)
        eligible = [t for t in eligible if af.get(t, 0.0) >= min_active_fraction]

    if price_floor > 0:
        last_px = close[eligible].ffill().iloc[-1]
        eligible = [t for t in eligible if last_px.get(t, 0.0) >= price_floor]

    if market_cap is not None and min_market_cap is not None:
        mc = market_cap.reindex(eligible)
        eligible = [t for t in eligible if pd.isna(mc[t]) or mc[t] >= min_market_cap]

    if not eligible:
        return []

    adv = avg_dollar_volume(close[eligible], volume[eligible], window,
                            fx=fx, cur_map=cur_map, as_of=as_of,
                            unit_factors=unit_factors, unknown=unknown)
    ranked = list(adv.sort_values(ascending=False).index)   # unknown currency dropped

    if len(ranked) <= topn:
        return [t for t in close.columns if t in set(ranked)]
    if not strata:
        return ranked[:topn]
    return _stratified_pick(ranked, topn, strata)
