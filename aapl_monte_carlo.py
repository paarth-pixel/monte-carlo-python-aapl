"""
Monte Carlo Simulation of AAPL Stock Price
------------------------------------------
Simulates future AAPL price paths using Geometric Brownian Motion (GBM):

    S_{t+dt} = S_t * exp[ (mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z ],  Z ~ N(0,1)

Drift (mu) and volatility (sigma) are estimated from historical daily
log returns pulled via yfinance. Falls back to manual parameters if the
data fetch fails (e.g. no internet).

Usage:
    pip install yfinance numpy matplotlib
    python aapl_monte_carlo.py
"""

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------- Parameters -----------------------------
TICKER        = "AAPL"
LOOKBACK      = "2y"      # history used to estimate mu and sigma
N_SIMS        = 10_000    # number of simulated paths
HORIZON_DAYS  = 252       # 1 trading year ahead
DT            = 1 / 252   # daily time step (in years)
SEED          = 42        # set to None for fresh randomness each run

# Fallback parameters if yfinance is unavailable
FALLBACK_S0    = 230.0    # spot price
FALLBACK_MU    = 0.10     # annualised drift
FALLBACK_SIGMA = 0.28     # annualised volatility


def fetch_market_params():
    """Pull historical prices and estimate S0, mu, sigma from log returns."""
    import yfinance as yf

    data = yf.Ticker(TICKER).history(period=LOOKBACK)["Close"].dropna()
    if len(data) < 30:
        raise ValueError("Not enough price history returned.")

    s0 = float(data.iloc[-1])
    log_ret = np.log(data / data.shift(1)).dropna()

    mu    = float(log_ret.mean()) * 252          # annualised drift
    sigma = float(log_ret.std(ddof=1)) * np.sqrt(252)  # annualised vol
    return s0, mu, sigma


def simulate_gbm(s0, mu, sigma, n_sims, n_days, dt, seed=None):
    """Vectorised GBM simulation. Returns array of shape (n_days+1, n_sims)."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_days, n_sims))

    drift     = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * z

    log_paths = np.vstack([np.zeros(n_sims), np.cumsum(drift + diffusion, axis=0)])
    return s0 * np.exp(log_paths)


def summarise(paths, s0, mu, sigma):
    """Print summary statistics of the terminal price distribution."""
    terminal = paths[-1]
    horizon_yrs = (paths.shape[0] - 1) * DT

    print(f"\n{'='*52}")
    print(f"  Monte Carlo Simulation — {TICKER}")
    print(f"{'='*52}")
    print(f"  Spot price (S0):          ${s0:,.2f}")
    print(f"  Annualised drift (mu):    {mu:.2%}")
    print(f"  Annualised vol (sigma):   {sigma:.2%}")
    print(f"  Paths simulated:          {paths.shape[1]:,}")
    print(f"  Horizon:                  {horizon_yrs:.2f} years")
    print(f"{'-'*52}")
    print(f"  Mean terminal price:      ${terminal.mean():,.2f}")
    print(f"  Median terminal price:    ${np.median(terminal):,.2f}")
    print(f"  Std dev:                  ${terminal.std(ddof=1):,.2f}")
    print(f"  5th percentile:           ${np.percentile(terminal, 5):,.2f}")
    print(f"  95th percentile:          ${np.percentile(terminal, 95):,.2f}")
    print(f"  P(terminal > S0):         {(terminal > s0).mean():.2%}")

    # 95% Value at Risk on simple returns over the horizon
    simple_ret = terminal / s0 - 1
    var_95 = np.percentile(simple_ret, 5)
    print(f"  95% VaR (horizon return): {var_95:.2%}")
    print(f"{'='*52}\n")


def plot_results(paths, s0):
    """Plot a sample of paths and the terminal price distribution."""
    terminal = paths[-1]
    days = np.arange(paths.shape[0])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                   gridspec_kw={"width_ratios": [2, 1]})

    # Left: sample of simulated paths + percentile bands
    n_show = min(200, paths.shape[1])
    ax1.plot(days, paths[:, :n_show], lw=0.5, alpha=0.25, color="steelblue")
    for q, style in [(5, "--"), (50, "-"), (95, "--")]:
        ax1.plot(days, np.percentile(paths, q, axis=1), style,
                 color="crimson", lw=1.8, label=f"{q}th percentile")
    ax1.axhline(s0, color="black", lw=1, ls=":", label="Spot")
    ax1.set_title(f"{TICKER} — {paths.shape[1]:,} GBM paths (showing {n_show})")
    ax1.set_xlabel("Trading days ahead")
    ax1.set_ylabel("Price ($)")
    ax1.legend()

    # Right: terminal distribution
    ax2.hist(terminal, bins=80, color="steelblue", edgecolor="white", alpha=0.85)
    ax2.axvline(s0, color="black", ls=":", lw=1.5, label="Spot")
    ax2.axvline(terminal.mean(), color="crimson", lw=1.5, label="Mean")
    ax2.set_title("Terminal price distribution")
    ax2.set_xlabel("Price ($)")
    ax2.set_ylabel("Frequency")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("aapl_monte_carlo.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    try:
        s0, mu, sigma = fetch_market_params()
        print(f"Fetched live data for {TICKER}.")
    except Exception as e:
        print(f"Data fetch failed ({e}). Using fallback parameters.")
        s0, mu, sigma = FALLBACK_S0, FALLBACK_MU, FALLBACK_SIGMA

    paths = simulate_gbm(s0, mu, sigma, N_SIMS, HORIZON_DAYS, DT, seed=SEED)
    summarise(paths, s0, mu, sigma)
    plot_results(paths, s0)
