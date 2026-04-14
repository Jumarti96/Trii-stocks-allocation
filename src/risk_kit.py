import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import datetime
import os
from scipy.stats import norm
from scipy.optimize import minimize
from numpy.linalg import inv

"""
LOAD TEST DATASETS FUNCTIONS
"""

# Function to load the Fama-French Dataset for the returns of the top and bottom deciles by MarketCap
def get_ffme_returns():
    """
    Load the Fama-French Dataset for the returns of the Top and Bottom deciles by MarketCap 
    """
    path = 'C:\Python projects\Finance\Jupyter notebooks and Python files\data\Portfolios_Formed_on_ME_monthly_EW.csv'
    me_m = pd.read_csv(path,
                       index_col = 0, header = 0, na_values = -99.99)
    portfolios = ['Lo 10', 'Hi 10']
    rets = me_m[portfolios]
    rets.columns = ['SmallCap', 'LargeCap']
    rets = rets/100
    rets.index = pd.to_datetime(rets.index, format = "%Y%m").to_period('M')
    return rets

# Function to load the EDHEC Hedge Fund Index Returns Dataset
def get_hfi_returns():
    """
    Load and format the EDHEC Hedge Fund Index Returns Dataset
    """
    path = 'C:\Python projects\Finance\Jupyter notebooks and Python files\data\edhec-hedgefundindices.csv'
    hfi = pd.read_csv(path,index_col = 0, header = 0, parse_dates = True)
    hfi = hfi/100
    hfi.index = hfi.index.to_period('M')
    return hfi

# Function to load the Ken French 30 Industry Portfolios Value Weighted Monthly Returns
def get_ind_file(filetype, weighting="vw", n_inds=30):
    """
    Load and format the Ken French Industry Portfolios files
    Variant is a tuple of (weighting, size) where:
        weighting is one of "ew", "vw"
        number of inds is 30 or 49
    """    
    if filetype == "returns":
        name = f"{weighting}_rets" 
        divisor = 100
    elif filetype == "nfirms":
        name = "nfirms"
        divisor = 1
    elif filetype == "size":
        name = "size"
        divisor = 1
    else:
        raise ValueError(f"filetype must be one of: returns, nfirms, size")
    
    ind = pd.read_csv(f"Jupyter notebooks and Python files/data/ind{n_inds}_m_{name}.csv", header=0, index_col=0, na_values=-99.99)/divisor
    ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind

def get_ind_returns(weighting="vw", n_inds=30):
    """
    Load and format the Ken French Industry Portfolios Monthly Returns
    """
    return get_ind_file("returns", weighting=weighting, n_inds=n_inds)

def get_ind_nfirms(n_inds=30):
    """
    Load and format the Ken French 30 Industry Portfolios Average number of Firms
    """
    return get_ind_file("nfirms", n_inds=n_inds)

def get_ind_size(n_inds=30):
    """
    Load and format the Ken French 30 Industry Portfolios Average size (market cap)
    """
    return get_ind_file("size", n_inds=n_inds)

def get_ind_market_caps(n_inds=30, weights=False):
    """
    Load the industry portfolio data and derive the market caps
    """
    ind_nfirms = get_ind_nfirms(n_inds=n_inds)
    ind_size = get_ind_size(n_inds=n_inds)
    ind_mktcap = ind_nfirms * ind_size
    if weights:
        total_mktcap = ind_mktcap.sum(axis=1)
        ind_capweight = ind_mktcap.divide(total_mktcap, axis="rows")
        return ind_capweight
    #else
    return ind_mktcap

def get_total_market_index_returns():
    """
    Calculate the total market index returns based on the Ken French 30 Industry Portfolio returns
    """
    ind_return = get_ind_returns()
    ind_size = get_ind_size()
    ind_nfirms = get_ind_nfirms()
    ind_mktcap = ind_nfirms*ind_size
    total_mktcap = ind_mktcap.sum(axis='columns')
    ind_capweight = ind_mktcap.divide(total_mktcap, axis='rows')
    total_market_return = (ind_capweight*ind_return).sum(axis='columns')
    # total_market_index = drawdown(total_market_return)['Wealth']
    return total_market_return

def get_fff_returns():
    path = "C:\Python projects\Finance\Jupyter notebooks and Python files\data\F-F_Research_Data_Factors_m.CSV"
    ind = pd.read_csv(path, index_col=0, header=0, parse_dates=True)
    ind = ind/100
    ind.index = pd.to_datetime(ind.index, format='%Y%m').to_period(freq='M')
    ind.columns = ind.columns.str.strip()
    return ind

# Drawdown function
def drawdown(return_series: pd.Series):
    """
    Takes a time series of asset returns
    Computes and returns a DataFrame that contains:
    a. The wealth index
    b. The previous peaks
    c. Percent drawdowns
    """
    wealth_index = 1000*(1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdown = (wealth_index - previous_peaks)/previous_peaks
    return pd.DataFrame({"Wealth": wealth_index,
                        "Peaks": previous_peaks,
                        "Drawdown": drawdown})

""" ********** SKEWNESS AND KURTOSIS COMPUTATIONS ********** """

# Skewness computation
def skewness(return_series: pd.Series):
    """
    Skewness calculation:
    S(returns) = mean((returns - mean(returns))**3)/stddev(returns)**3
    Take the mean of the deviations from the mean at the cubic power and divide by the volatility at the cubic power
    """
    mean_returns = return_series.mean()
    demeaned_returns = return_series - mean_returns
    var_returns = (demeaned_returns**2).sum()/len(return_series)
    std_returns = var_returns**(1/2)
    return (demeaned_returns**3).mean()/std_returns**3


# Kurtosis computation
def kurtosis(return_series: pd.Series):
    """
    Kurtosis calculation:
    K(returns) = mean((returns - mean(returns))**4)/stddev(returns)**4
    Take the mean of the deviations from the mean at the 4th power and divide by the volatility at the 4th power
    """
    mean_returns = return_series.mean()
    demeaned_returns = return_series - mean_returns
    var_returns = (demeaned_returns**2).sum()/len(return_series)
    std_returns = var_returns**(1/2)
    return (demeaned_returns**4).mean()/std_returns**4

# Jarque-Bera test of normality computation
def jb_test(return_series: pd.Series):
    """
    Jarque-Bera test of normality:
    JB = (n/6)*(S**2 + ((K-3)**2)/4) -> Chi-squared(2)    
    """
    S = skewness(return_series)
    K = kurtosis(return_series)
    n = len(return_series)
    return (n/6)*(S**2 + ((K-3)**2)/4)

# Test of normality
def is_normal(return_series, level=0.01):
    if isinstance(return_series, pd.DataFrame):
        return return_series.apply(stats.jarque_bera, axis=0).iloc[1]  > 0.05
    elif isinstance(return_series, pd.Series) or isinstance(return_series, np.ndarray):
        return stats.jarque_bera(return_series)[1] > 0.05
    else:
        return TypeError("Expected return_series to be Series or DataFrame.")

"""
********** SOME MATH TRANSFORMATIONS **********
"""
def compound(return_series):
    """
    Returns the compound returns of a series of returns
    """
    return (1 + return_series).prod() - 1

def annualize_vol(return_series, periods_per_year = 12):
    """
    Annualizes the vol of a set of returns
    """
    return return_series.std(ddof=0)*(periods_per_year**(1/2))

def annualize_returns(return_series, periods_per_year = 12):
    """
    Annualizes the returns of a set of returns
    """
    compounded_rets = (1 + return_series).prod()
    avg_rets = compounded_rets**(1/len(return_series)) - 1
    return (1 + avg_rets)**periods_per_year - 1

def sharpe_ratio(return_series, riskfree_rate, periods_per_year = 12):
    """
    Calculates the sharpe ratio of a set of returns
    """
    rf_per_period = (1 + riskfree_rate)**(1/periods_per_year) - 1
    excess_rets = return_series - rf_per_period
    annualized_returns = annualize_returns(excess_rets, periods_per_year=periods_per_year)
    annualized_vol = annualize_vol(excess_rets, periods_per_year=periods_per_year)
    return annualized_returns/annualized_vol

def weight_ew(r, cap_weights=None, max_cw_mult=None, microcap_threshold=None, **kwargs):
    """
    Returns the weights of the equally weighted portfolio on the asset returns "r" as a DataFrame
    """
    n = len(r.columns)
    ew = pd.Series(1/n, index=r.columns)
    if cap_weights is not None:
        cw = cap_weights.loc[r.index[0]]    # Starting cap weight
        ## exclude microcaps
        if microcap_threshold is not None and microcap_threshold > 0:
            microcap = cw < microcap_threshold
            ew[microcap] = 0
            ew = ew/ew.sum()
        if max_cw_mult is not None:
            cw = np.minimum(cw, max_cw_mult*ew)
            ew = ew/ew.sum()
    return ew

def weight_cw(r, cap_weights, **kwargs):
    """
    Returns the weights of the cap-weighted portfolio on the asset returns "r" as a DataFrame
    """
    return cap_weights.loc[r.index[0]]

"""
********** EXTREME RISK CALCULATION METHODS **********
"""
# Semideviation
def semideviation(return_series, cutoff_criteria = "below-zero"):
    """
    Return the semideviation, which is the volatility (standard deviation) of below-zero or below-mean returns
    cutoff_criteria can be either:
    "below-zero" to obtain the volatility of below-zero returns
    "below-mean" to obtain the volatility of below-mean returns
    """
    valid_criteria = {"below-zero", "below-mean"}
    if cutoff_criteria == "below-zero":
        is_negative_return = return_series < 0
        return return_series[is_negative_return].std(ddof=0)
    elif cutoff_criteria == "below-mean":
        demeaned_returns = return_series - return_series.mean()
        is_negative_return = demeaned_returns < 0
        return return_series[is_negative_return].std(ddof=0)
    else:
        raise ValueError("results: cutoff_criteria must be one of %r." %valid_criteria)

# VaR based on historical returns
def var_historic(return_series, level=5):
    if isinstance(return_series, pd.DataFrame):
        return return_series.aggregate(var_historic, level=level)
    elif isinstance(return_series, pd.Series) or isinstance(return_series, np.ndarray):
        return -np.percentile(return_series, q=level)
    else:
        raise TypeError("Expected return_series to be Series or DataFrame.")
    
# CVaR based on historical returns
def cvar_historic(return_series, level=5):
        is_beyond = return_series <= -var_historic(return_series, level=level)
        return -return_series[is_beyond].mean()
    
# Gaussian VaR
def var_gaussian(return_series, level=5, modified=False):
    """
    Returns the Parametric Gaussian VaR of a Series or DataFrame
    If "modified" is True, then the modified VaR is returned,
    using the Cornish-Fisher modification
    """
    z = norm.ppf(level/100)
    if modified == True:
        S = skewness(return_series)
        K = kurtosis(return_series)
        z = (z + 
                (1/6)*(z**2 - 1)*S + 
                (1/24)*(z**3 - 3*z)*(K - 3) - 
                (1/36)*(2*z**3 - 5*z)*S**2
            )
    return -(return_series.mean() + z*return_series.std(ddof=0))

"""
********** COVARIANCE ESTIMATORS **********
"""
# Function to calculate the sample covariance of the supplied returns
def sample_cov(r, **kwargs):
    """
    Returns the sample covariance of the supplied returns.
    """
    return r.cov()

# Function to estimate a covariance matrix by using the Elton/Gruber Constant Correlation model
def cc_cov(r, **kwargs):
    """
    Estimates a covariance matrix by using the Elton/Gruber Constant Correlation model
    """
    rhos = r.corr()
    n = rhos.shape[0]
    # this is a symmetric matrix with diagonals all 1 - so the mean correlation is ...
    rho_bar = (rhos.values.sum()-n)/(n*(n-1))
    ccor = np.full_like(rhos, rho_bar)
    np.fill_diagonal(ccor, 1.)
    sd = r.std()
    return pd.DataFrame(ccor * np.outer(sd, sd), index=r.columns, columns=r.columns)

def shrinkage_cov(r, delta=0.5, **kwargs):
    """
    Covariance estimator that shrinks between the Sample Covariance and the Constant Correlation Estimators
    """
    prior = cc_cov(r, **kwargs)
    sample = sample_cov(r, **kwargs)
    return delta*prior + (1-delta)*sample

def returns_covmat_validation(**kwargs):
    """
    Returns the appropriate returns and covariance matrix based on the inputs received
    return_series: a dataframe with 2 columns, each corresponding to an asset\n
    returns: an array of length 2 with returns for 2 assets\n
    covmat: a covariance matrix of shape 2 by 2 for 2 assets\n
    periods_per_year: an integer for the number of periods per year for returns annualization\n
    """
    return_series = kwargs.get('return_series', pd.DataFrame())
    returns = kwargs.get('returns', None)
    covmat = kwargs.get('covmat', None)
    periods_per_year = kwargs.get('periods_per_year', 12)
    if return_series.empty and ((returns is None) or (covmat is None)):
        raise ValueError("Missing returns data")
    if not return_series.empty:
        returns = annualize_returns(return_series, periods_per_year)
        covmat = return_series.cov()
    return returns, covmat

"""
********** ASSET EFFICIENT FRONTIER **********
"""
# Function to calculate the expected return of a combination of portfolios based on a given set of weights
def portfolio_return(weights, return_series):
    """
    Weights -> Returns
    """
    return weights.T @ return_series

# Function to calculate the expected volatility of a combination of portfolios based on a given set of weights
def portfolio_vol(weights, covmat):
    """
    Weights -> Vol
    """
    return (weights.T @ covmat @ weights)**(1/2)

# Define a function to find the combination with the less volatility for a desired level of returns
def minimize_vol(target_return, **kwargs):
    """
    target_ret -> W\n
    Possible data inputs:\n
    return_series: a dataframe with 2 columns, each corresponding to an asset\n
    returns: an array of length 2 with returns for 2 assets\n
    covmat: a covariance matrix of shape 2 by 2 for 2 assets\n
    periods_per_year: an integer for the number of periods per year for returns annualization\n
    """
    returns, covmat = returns_covmat_validation(**kwargs)
    n = len(returns)
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),)*n
    return_is_target = {
        'type': 'eq',
        'args': (returns,),
        'fun': lambda weights, returns: target_return - portfolio_return(weights, returns)
    }
    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }
    results = minimize(portfolio_vol, init_guess,
                       args = (covmat,), method = 'SLSQP',
                       options = {'disp': False},
                       constraints = (return_is_target, weights_sum_to_1),
                       bounds=bounds
                       )
    return results.x

def optimal_weights(n_points, **kwargs):
    """
    List of weights to run the optimizer on to minimize the volatility\n
    Possible data inputs:\n
    return_series: a dataframe with 2 columns, each corresponding to an asset\n
    returns: an array of length 2 with returns for 2 assets\n
    covmat: a covariance matrix of shape 2 by 2 for 2 assets\n
    periods_per_year: an integer for the number of periods per year for returns annualization\n
    """
    returns, covmat = returns_covmat_validation(**kwargs)
    target_returns = np.linspace(np.min(returns), np.max(returns), n_points)
    return [minimize_vol(r, returns = returns, covmat = covmat) for r in target_returns]

# Define a function to find the combination with the highest Sharpe-Ratio
def msr(riskfree_rate, **kwargs):
    """
    RiskFree rate + Returns + Covariance -> W\n
    Possible data inputs:\n
    return_series: a dataframe with 2 columns, each corresponding to an asset\n
    returns: an array of length 2 with returns for 2 assets\n
    covmat: a covariance matrix of shape 2 by 2 for 2 assets\n
    periods_per_year: an integer for the number of periods per year for returns annualization\n
    """
    returns, covmat = returns_covmat_validation(**kwargs)
    n = returns.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),)*n
    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }
    def neg_sharpe_ratio(weights, riskfree_rate, returns, covmat):
        """
        Calculates the negative of the Sharpe-Ratio for a given combination of returns, covariances and weights.
        """
        ret = portfolio_return(weights, returns)
        vol = portfolio_vol(weights, covmat)
        return -(ret - riskfree_rate)/vol

    def neg_sharpe_gradient(weights, riskfree_rate, returns, covmat):
        """
        Calculates the analytical gradient explicitly to massively speed up SciPy's SLSQP.
        """
        ret = portfolio_return(weights, returns)
        vol = portfolio_vol(weights, covmat)
        
        R = returns.values if hasattr(returns, 'values') else np.array(returns)
        C = covmat.values if hasattr(covmat, 'values') else np.array(covmat)
        
        # Quotient Rule derivative w.r.t weights (w):
        # u = ret - rf, u' = R
        # v = vol, v' = (C @ w) / vol
        dv = (C @ weights) / vol
        grad = -(R * vol - (ret - riskfree_rate) * dv) / (vol**2)
        return grad

    results = minimize(neg_sharpe_ratio, init_guess,
                       args = (riskfree_rate, returns, covmat), method = 'SLSQP',
                       # jac = neg_sharpe_gradient,  # DISABLED
                       options = {'disp': False},
                       constraints = (weights_sum_to_1),
                       bounds=bounds
                       )
    return results.x

# Define a function to find the combination with the highest Sharpe-Ratio, with a few adjustments
def msr_tuned(riskfree_rate, max_weight=1.0, **kwargs):
    """
    Returns the Sharpe-ratio maximizing portfolio, with a parameter to set a minimum acceptable weight:\n
    RiskFree rate + Returns + Covariance -> W\n
    Possible data inputs:\n
    return_series: a dataframe with 2 columns, each corresponding to an asset\n
    returns: an array of length 2 with returns for 2 assets\n
    covmat: a covariance matrix of shape 2 by 2 for 2 assets\n
    periods_per_year: an integer for the number of periods per year for returns annualization\n
    debug: if True, prints optimizer progress and convergence info (default: False)\n
    """
    debug = kwargs.pop('debug', False)
    returns, covmat = returns_covmat_validation(**kwargs)
    n = returns.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, max_weight),)*n
    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }
    def neg_sharpe_ratio(weights, riskfree_rate, returns, covmat):
        """
        Calculates the negative of the Sharpe-Ratio for a given combination of returns, covariances and weights.
        """
        ret = portfolio_return(weights, returns)
        vol = portfolio_vol(weights, covmat)
        return -(ret - riskfree_rate)/vol

    #TODO: Pendiente a revisar 
    def neg_sharpe_gradient(weights, riskfree_rate, returns, covmat):
        """
        Calculates the analytical gradient explicitly to massively speed up SciPy's SLSQP.
        """
        ret = portfolio_return(weights, returns)
        vol = portfolio_vol(weights, covmat)
        
        R = returns.values if hasattr(returns, 'values') else np.array(returns)
        C = covmat.values if hasattr(covmat, 'values') else np.array(covmat)
        
        # Quotient Rule derivative w.r.t weights (w):
        dv = (C @ weights) / vol
        grad = -(R * vol - (ret - riskfree_rate) * dv) / (vol**2)
        return grad

    # [3] Callback: called after every iteration when debug=True
    iteration_counter = [0]
    def optimizer_callback(weights):
        iteration_counter[0] += 1
        ret = portfolio_return(weights, returns)
        vol = portfolio_vol(weights, covmat)
        sharpe = (ret - riskfree_rate) / vol
        print(f"  [iter {iteration_counter[0]:>3}] Sharpe: {sharpe:.6f} | Ret: {ret:.4f} | Vol: {vol:.4f}")

    results = minimize(neg_sharpe_ratio, init_guess,
                       args = (riskfree_rate, returns, covmat),
                       method = 'SLSQP',
                       # jac = neg_sharpe_gradient,  # DISABLED
                       # [1] disp: print summary on completion   [2] iprint: print each iteration
                       options = {'disp': debug, 'iprint': 2 if debug else -1},
                       callback = optimizer_callback if debug else None,
                       constraints = (weights_sum_to_1),
                       bounds=bounds
                       )

    # [4] Inspect results object for silent failures
    if debug:
        if results.success:
            print(f"   Optimizer converged in {results.nit} iterations ({results.nfev} function evaluations).")
        else:
            print(f"    Optimizer did NOT converge: {results.message}")
            print(f"      Iterations used: {results.nit} | Function evaluations: {results.nfev}")

    return results.x

def gmv(covmat):
    """
    Returns the weight of the Global Minimum Volatility portfolio
    given the covariance matrix
    """
    n_assets = covmat.shape[0]
    return msr(0, returns=np.repeat(1, n_assets), covmat=covmat)

# Function to plot the Efficient Frontier for a combination of 2 assets
def plot_ef2(n_points, style = '.-', **kwargs):
    """
    Plot the Efficient Frontier for a combination of 2 assets\n
    Possible data inputs:\n
    return_series: a dataframe with 2 columns, each corresponding to an asset\n
    returns: an array of length 2 with returns for 2 assets\n
    covmat: a covariance matrix of shape 2 by 2 for 2 assets\n
    periods_per_year: an integer for the number of periods per year for returns annualization\n
    """
    returns, covmat = returns_covmat_validation(**kwargs)
    weights = [np.array([w, 1-w]) for w in np.linspace(0, 1, n_points)]
    rets = [portfolio_return(w, returns) for w in weights]
    vols = [portfolio_vol(w, covmat) for w in weights]
    ef = pd.DataFrame({"Return": rets, "Volatility": vols})
    ax = ef.plot.line(x="Volatility", y="Return", style=style)
    return ax

# Function to plot the Efficient Frontier for a combination of 2 assets
def plot_ef(n_points, style = '.-', show_cml=False, show_ew=False, show_gmv=False, riskfree_rate=0, **kwargs):
    """
    Plot the N-asset Efficient Frontier\n
    Possible data inputs:\n
    return_series: a dataframe with columns for multiple assets\n
    returns: an array of length 2 with returns for 2 assets\n
    covmat: a covariance matrix of shape 2 by 2 for 2 assets\n
    periods_per_year: an integer for the number of periods per year for returns annualization\n
    """
    returns, covmat = returns_covmat_validation(**kwargs)
    periods_per_year = kwargs.get('periods_per_year', 12)
    weights = optimal_weights(n_points, returns = returns, covmat = covmat)
    rets = [portfolio_return(w, returns) for w in weights]
    vols = [portfolio_vol(w, covmat) for w in weights]
    ef = pd.DataFrame({"Return": rets, "Volatility": vols})
    ax = ef.plot.line(x="Volatility", y="Return", style=style)
    if show_cml:
        w_msr = msr(riskfree_rate, returns=returns, covmat=covmat, periods_per_year=periods_per_year)
        returns_msr = portfolio_return(w_msr, returns)
        volatility_msr = portfolio_vol(w_msr, covmat)
        cml_x = [0, volatility_msr]
        cml_y = [riskfree_rate, returns_msr]
        ax.plot(cml_x, cml_y, color='green', marker='o', linestyle='dashed',  label='Maximum Sharpe-Ratio Allocation')
    if show_ew:
        n_assets = len(returns)
        w_ew = np.repeat(1/n_assets, n_assets)
        returns_ew = portfolio_return(w_ew, returns)
        volatility_ew = portfolio_vol(w_ew, covmat)
        ew_x = volatility_ew
        ew_y = returns_ew
        ax.plot(ew_x, ew_y, color='goldenrod', marker='o', label='Equal Weights Allocation')
    if show_gmv:
        w_gmv = gmv(covmat)
        returns_gmv = portfolio_return(w_gmv, returns)
        volatility_gmv = portfolio_vol(w_gmv, covmat)
        ax.plot(volatility_gmv, returns_gmv, color='purple', marker='o', label='Global Minimum Volatility')
    ax.legend()
    return ax

# Function to find the weights of the GMV portfolio given a covariance matrix of the returns
def weight_gmv(r, cov_estimator=sample_cov, **kwargs):
    """
    Produces the weights of the GMV portfolio given a covariance matrix of the returns
    """
    est_cov = cov_estimator(r, **kwargs)
    return gmv(est_cov)

"""
********** CPPI (Constant Proportion Portfolio Insurance) **********
"""
def run_cppi(risky_r, safe_r=None, m=3, start=1000, floor=0.8, riskfree_rate=0.03, drawdown=None):
    """
    Run a backtest of the CPPI strategy, given a set of returns for the risky asset
    Returns a dictionary containing: Asset Value History, Risk Budget History, Risky Weight History
    """
    # set up the CPPI parameters
    dates = risky_r.index
    n_steps = len(dates)
    account_value = start
    floor_value = start*floor
    peak = start

    if isinstance(risky_r, pd.Series):
        risky_r = pd.DataFrame(risky_r)
        risky_r.columns = ['R']
    
    if safe_r is None:
        safe_r = pd.DataFrame().reindex_like(risky_r)
        safe_r.values[:] = riskfree_rate/12 # fast way to set all values to a number
    # set up some dataframes to save historical values
    account_history = pd.DataFrame().reindex_like(risky_r)
    risky_w_history = pd.DataFrame().reindex_like(risky_r)
    cushion_history = pd.DataFrame().reindex_like(risky_r)
    floor_v_history = pd.DataFrame().reindex_like(risky_r)

    for step in range(n_steps):
        if drawdown is not None:
            peak = np.maximum(peak, account_value)
            floor_value = peak*(1-drawdown)
        cushion = (account_value - floor_value)/account_value
        risky_w = cushion*m
        risky_w = np.minimum(risky_w, 1)
        risky_w = np.maximum(risky_w, 0)
        safe_w = 1 - risky_w
        risky_allocation = account_value*risky_w
        safe_allocation = account_value*safe_w
        account_value = risky_allocation*(1+risky_r.iloc[step]) + safe_allocation*(1+safe_r.iloc[step])
        # Save the values so we can look at the history and plot
        account_history.iloc[step] = account_value
        risky_w_history.iloc[step] = risky_w
        cushion_history.iloc[step] = cushion
        floor_v_history.iloc[step] = floor_value

    risky_wealth = start*(1+risky_r).cumprod()
    backtest_result = {
        "Wealth": account_history,
        "Risky Wealth": risky_wealth,
        "Risk Budget": cushion_history,
        "Risky Allocation": risky_w_history,
        "m": m,
        "start": start,
        "floor": floor_v_history,
        "risky_r": risky_r,
        "safe_r": safe_r
    }
    return backtest_result

def summary_stats(r, riskfree_rate=0.0, periods_per_year=12, level=5):
    """
    Return a dataframe that contains aggregated summary stats for the returns in the columns of r
    """
    ann_r = r.aggregate(annualize_returns, periods_per_year=periods_per_year)
    ann_vol = r.aggregate(annualize_vol, periods_per_year=periods_per_year)
    ann_sr = r.aggregate(sharpe_ratio, riskfree_rate=riskfree_rate, periods_per_year=periods_per_year)
    dd = r.aggregate(lambda r: drawdown(r).Drawdown.min())
    skew = r.aggregate(skewness)
    kurt = r.aggregate(kurtosis)
    cf_var = r.aggregate(var_gaussian, modified=True, level=level)
    hist_cvar = r.aggregate(cvar_historic, level=level)
    return pd.DataFrame({
        "Annualized Return": ann_r,
        "Annualized Volatility": ann_vol,
        "Skewness": skew,
        "Kurtosis": kurt,
        f"Cornish-Fisher VaR ({level}%)": cf_var,
        f"Historic CVaR ({level}%)": hist_cvar,
        "Sharpe Ratio": ann_sr,
        "Max Drawdown": dd
    })

"""
********** RANDOM WALKS AND MONTE CARLO SIMULATIONS **********
"""
def gbm(n_years=10, steps_per_year=12, n_scenarios=1000, mu=0.07, sigma=0.15, s_0=100.0, prices=True):
    """
    Evolution of a Stock Price using a Geometric Brownian Motion Model
    """
    dt = 1/steps_per_year
    n_steps = int(n_years*steps_per_year) + 1
    rets_plus_1 = np.random.normal(loc=1+mu*dt, scale=sigma*np.sqrt(dt), size=(n_steps, n_scenarios))
    rets_plus_1[0] = 1
    # convert into prices
    rets_val = s_0*pd.DataFrame(rets_plus_1).cumprod() if prices else rets_plus_1-1
    return rets_val

"""
********** VISUALIZATION OF STRATEGIES **********
"""

def show_gbm(n_scenarios, mu, sigma):
    """
    Draw the results of a stock price evolution under a Geometric Brownian Motion model
    """
    s_0 = 100
    prices = gbm(n_scenarios=n_scenarios, mu=mu, sigma=sigma, s_0=s_0)
    ax = prices.plot(legend=False, color='indianred', alpha=0.5, linewidth=2, figsize=(12, 5))
    ax.axhline(y=s_0, ls=':', color='black')
    # draw a dot at the origin
    ax.plot(0, s_0, marker='o', color='darkred', alpha=0.2)

def show_cppi(n_scenarios=50, n_years=10, steps_per_year=12, mu=0.07, sigma=0.15, m=3, floor=0., riskfree_rate=0.03, y_max=100):
    """
    Plot the results of a Monte Carlo simulation of CPPI
    """
    start = 100
    sim_rets = gbm(n_scenarios=n_scenarios, n_years=n_years, steps_per_year=steps_per_year, mu=mu, sigma=sigma, prices=False)
    risky_r = pd.DataFrame(sim_rets)
    # run the "back"-test
    btr = run_cppi(risky_r=pd.DataFrame(risky_r), riskfree_rate=riskfree_rate, m=m, start=start, floor=floor)
    wealth = btr['Wealth']

    # calculate terminal wealth stats
    y_max = wealth.values.max()*y_max/100
    terminal_wealth = wealth.iloc[-1]

    tw_mean = terminal_wealth.mean()
    tw_median = terminal_wealth.median()
    failure_mask = np.less(terminal_wealth, start*floor)
    n_failures = failure_mask.sum()
    p_fail = n_failures/n_scenarios

    e_shortfall = np.dot(terminal_wealth-start*floor, failure_mask)/n_failures if n_failures > 0 else 0.0

    # Plots
    fig, (wealth_ax, hist_ax) = plt.subplots(nrows=1, ncols=2, sharey=True, gridspec_kw={'width_ratios':[3, 2]}, figsize=(24, 9))
    plt.subplots_adjust(wspace=0.0)

    wealth.plot(ax=wealth_ax, legend=False, alpha=0.3, color='indianred')
    wealth_ax.axhline(y=start, ls=':', color='black')
    wealth_ax.axhline(y=start*floor, ls='--', color='red')
    wealth_ax.set_ylim(top=y_max)

    terminal_wealth.plot.hist(ax=hist_ax, bins=50, ec='w', fc='indianred', orientation='horizontal')
    hist_ax.axhline(y=start, ls=':', color='black')
    hist_ax.axhline(y=tw_mean, ls=':', color='blue')
    hist_ax.axhline(y=tw_median, ls=':', color='purple')
    hist_ax.annotate(f'Mean: ${int(tw_mean)}', xy=(.5, .9), xycoords='axes fraction', fontsize=24)
    hist_ax.annotate(f'Median: ${int(tw_median)}', xy=(.5, .85), xycoords='axes fraction', fontsize=24)
    if (floor > 0.01):
        hist_ax.axhline(y=start*floor, ls='--', color='red', linewidth=3)
        hist_ax.annotate(f'Violations: {n_failures} ({p_fail*100:2.2f}%)\nE(shortfall) = ${e_shortfall:2.2f}',
                         xy=(.5, .7), xycoords='axes fraction', fontsize=24)
        
"""
********** FUNDING RATIO, PRESENT VALUE, AND PRICE DISCOUNT **********
"""
def discount(r, t):
    """
    Compute the price of a pure discount bond that pays a dollar at time t,
    given a per'period interest rate r
    returns a |t| x |r| Series or DataFrame
    r can be a float, Series or DataFrame
    returns a DataFrame indexed by t
    """
    discounts = pd.DataFrame([(r + 1)**-i for i in t])
    discounts.index = t
    return discounts

def pv(flows, r):
    """
    Compute the present value of a sequence of cash flows given by the time (as an index) and amounts
    r can be a scalar, or a Series or DataFrame with the number of rows matching the num of rows in flows
    """
    dates = flows.index
    discounts = discount(r, dates)
    return discounts.multiply(flows, axis='rows').sum()

def funding_ratio(assets, liabilities, r):
    """
    Computes the funding ratio of some assets given liabilities and interest rate
    """
    return pv(assets, r)/pv(liabilities, r)

"""
********** CIR (COX INGERSOLL ROSS) MODEL OF DYNAMIC INTEREST RATES **********
"""
def instant_to_annual(r):
    """
    Converts short rate to an annualized rate
    """
    return np.expm1(r)

def annual_to_instant(r):
    """
    Converts annualized to a short rate
    """
    return np.log1p(r)

def cir(n_years=10, n_scenarios=1, a=.05, b=.03, sigma=.05, steps_per_year=12, r_0=None):
    """
    Implements the CIR model for interest rates
    Generates random interest rate evolution over time using the CIR model
    b and r_0 are assumed to be the annualized rates, not the short rates
    and the returned values are the annualized rates as well
    """
    if r_0 is None: r_0 = b 
    r_0 = annual_to_instant(r_0)
    dt = 1/steps_per_year
    num_steps = int(n_years*steps_per_year) + 1 # because n_years might be a float
    
    shock = np.random.normal(0, scale=np.sqrt(dt), size=(num_steps, n_scenarios))
    rates = np.empty_like(shock)
    rates[0] = r_0

    ## For Price Generation
    h = np.sqrt(a**2 + 2*sigma**2)
    prices = np.empty_like(shock)
    ####

    def price(ttm, r):
        _A = ((2*h*np.exp((h+a)*ttm/2))/(2*h+(h+a)*(np.exp(h*ttm)-1)))**(2*a*b/sigma**2)
        _B = (2*(np.exp(h*ttm)-1))/(2*h + (h+a)*(np.exp(h*ttm)-1))
        _P = _A*np.exp(-_B*r)
        return _P
    prices[0] = price(n_years, r_0)
    ####
    
    for step in range(1, num_steps):
        r_t = rates[step-1]
        d_r_t = a*(b-r_t)*dt + sigma*np.sqrt(r_t)*shock[step]
        rates[step] = abs(r_t + d_r_t)
        # generate prices at time t as well ...
        prices[step] = price(n_years-step*dt, rates[step])

    rates = pd.DataFrame(data=instant_to_annual(rates), index=range(num_steps))
    ### for prices
    prices = pd.DataFrame(data=prices, index=range(num_steps))
    ###
    return rates, prices

def show_cir(n_years=10, n_scenarios=1, a=.05, b=.03, sigma=.05, steps_per_year=12, r_0=None, prices=False):
    option = 1 if prices else 0
    cir(n_years=n_years,
        n_scenarios=n_scenarios, 
        a=a, 
        b=b, 
        sigma=sigma, 
        steps_per_year=steps_per_year, 
        r_0=r_0
        )[option].plot(legend=False, figsize=(12, 5))
    
"""
********** LIABILITY DRIVEN INVESTMENT (LDI) **********
"""
def bond_cash_flows(maturity, principal=100, coupon_rate=.03, coupons_per_year=12):
    """
    Returns a series of cash flows generated by a bond,
    indexed by a coupon number
    """
    n_coupons = round(maturity*coupons_per_year)
    coupon_amt = principal*coupon_rate/coupons_per_year
    coupons = np.repeat(coupon_amt, n_coupons)
    coupon_times = np.arange(1, n_coupons+1)
    cash_flows = pd.Series(data=coupon_amt, index=coupon_times)
    cash_flows.iloc[-1] += principal
    return cash_flows

def bond_price(maturity, principal=100, coupon_rate=.03, coupons_per_year=12, discount_rate=.03):
    """
    Price a bond based on bond parameters maturity, principal, coupon rate, coupons per year
    and the prevailing discount rate
    """
    if isinstance(discount_rate, pd.DataFrame):
        pricing_dates = discount_rate.index
        prices = pd.DataFrame(index=pricing_dates, columns=discount_rate.columns)
        for t in pricing_dates:
            prices.loc[t] = bond_price(maturity-t/coupons_per_year, principal, coupon_rate, coupons_per_year,
                                      discount_rate.loc[t])
        return prices
    else: # base case ... single time period
        if maturity <= 0: return principal+principal*coupon_rate/coupons_per_year
        cash_flows = bond_cash_flows(maturity, principal, coupon_rate, coupons_per_year)
        return pv(cash_flows, discount_rate/coupons_per_year)

def bond_total_return(monthly_prices, principal, coupon_rate, coupons_per_year):
    """
    Computes the total return of a bond based on monthly bond prices and coupon payments
    Assumes that dividens (coupons) are paid out at the end of the period (e.g. end of 3 months for quarterly div)
    and that dividends are reinvested in the bond
    """
    coupons = pd.DataFrame(data=0, index=monthly_prices.index, columns=monthly_prices.columns)
    t_max = monthly_prices.index.max()
    pay_date = np.linspace(12/coupons_per_year, t_max, int(coupons_per_year*t_max/12), dtype=int)
    coupons.iloc[pay_date] = principal*coupon_rate/coupons_per_year
    total_returns = (monthly_prices + coupons)/monthly_prices.shift()-1
    return total_returns.dropna()

def macaulay_duration(flows, discount_rate):
    """
    Computes the Macaulay Durtation of a sequence of cash flows
    """
    discounts = discount(discount_rate, flows.index)
    dcf = discounts*pd.DataFrame(flows)
    weights = dcf/dcf.sum()
    # return np.average(flows.index, weights=weights.iloc[:,0])
    return np.dot(flows.index, weights)

def match_durations(cf_t, cf_s, cf_l, discount_rate):
    """
    Returns the weight W in cf_s that, along with (1 - W) in cf_l will have an effective duration
    that matches cf_t.\n
    Where:\n
    cf_t: target cash flows\n
    cf_s: cash flows of the short-term bond\n
    cf_l: cash flows of the long-term bond
    """
    d_t = macaulay_duration(cf_t, discount_rate)
    d_s = macaulay_duration(cf_s, discount_rate)
    d_l = macaulay_duration(cf_l, discount_rate)
    return (d_l - d_t)/(d_l - d_s)

def bt_mix(r1, r2, allocator, **kwargs):
    """
    Runs a back test (simulation) of allocating between a two sets of return
    r1 and 42 are T x N DataFrames or returns where T is the time step index and N is the number of scenarios.
    allocator is a function that takes two sets of returns and allocator specific parameters, and produces
    an allocation to the first portfolio (the rest of the money is invested in the GHP) as a T x 1 DataFrame
    Returns a T x N DataFrame of the resulting N portfolio scenarios
    """
    if not r1.shape == r2.shape:
        raise ValueError("r1 and r2 need to be the same shape.")
    weights = allocator(r1, r2, **kwargs)
    if not weights.shape == r1.shape:
        raise ValueError("Allocator returned weights that don't match r1.")
    r_mix = weights*r1 + (1-weights)*r2
    return r_mix

def fixedmix_allocator(r1, r2, w1, **kwargs):
    """
    Produces a time series over T steps of allocations between the PSP and GHP across N scenarios
    PSP and GHP are T x N DataFrames that represent the returns of the PSP and GHP such that:
        each column is a scenario
        each row is the price for a timestep
    Returns an T x N DataFrame of PSP Weights
    """
    return pd.DataFrame(data=w1, index=r1.index, columns=r1.columns)

def glidepath_allocator(r1, r2, start_glide = 1, end_glide = 0):
    """
    Simulate a Target-Date-Fund style gradual move from r1 to r2
    """
    n_points = r1.shape[0]
    n_col = r1.shape[1]
    path = pd.Series(data=np.linspace(start_glide, end_glide, num=n_points))
    paths = pd.concat([path]*n_col, axis='columns')
    paths.index = r1.index
    paths.columns = r1.columns
    return paths

def floor_allocator(psp_r, ghp_r, floor, zc_prices, m=3):
    """
    Allocate between PSP and GHP with the goal to provide exposure to the upside
    of the PSP without going violating the floor.
    Uses a CPPI-style dynamic risk budgeting algorithm by investing a multiple
    of the cushion in the PSP
    Returns a DataFrame with the same shape as the psp/ghp representing the weights in the PSP
    """
    if zc_prices.shape != psp_r.shape:
        raise ValueError('PSP and ZC Prices must have the same shape.')
    n_steps, n_scenarios = psp_r.shape
    account_value = np.repeat(1, n_scenarios)
    floor_value = np.repeat(1, n_scenarios)
    w_history = pd.DataFrame(index=psp_r.index, columns=psp_r.columns)
    for step in range(n_steps):
        floor_value = floor*zc_prices.iloc[step] ## PV of Floor assuming today's rates and flat YC
        cushion = (account_value - floor_value)/account_value
        psp_w = (m*cushion).clip(0, 1)  #same as applying min and max
        ghp_w = 1-psp_w
        psp_alloc = account_value*psp_w
        ghp_alloc = account_value*ghp_w
        # recompute the new account value at the end of this step
        account_value = psp_alloc*(1+psp_r.iloc[step]) + ghp_alloc*(1+ghp_r.iloc[step])
        w_history.iloc[step] = psp_w
    return w_history

def drawdown_allocator(psp_r, ghp_r, maxdd, m=3):
    """
    Allocate between PSP and GHP with the goal to provide exposure to the upside
    of the PSP without going violating the floor.
    Uses a CPPI-style dynamic risk budgeting algorithm by investing a multiple
    of the cushion in the PSP
    Returns a DataFrame with the same shape as the psp/ghp representing the weights in the PSP
    """
    n_steps, n_scenarios = psp_r.shape
    account_value = np.repeat(1, n_scenarios)
    floor_value = np.repeat(1, n_scenarios)
    peak_value = np.repeat(1, n_scenarios)
    w_history = pd.DataFrame(index=psp_r.index, columns=psp_r.columns)
    for step in range(n_steps):
        floor_value = (1-maxdd)*peak_value ### Floor is based on Prev Peak
        cushion = (account_value - floor_value)/account_value
        psp_w = (m*cushion).clip(0, 1)  #same as applying min and max
        ghp_w = 1-psp_w
        psp_alloc = account_value*psp_w
        ghp_alloc = account_value*ghp_w
        # recompute the new account value at the end of this step
        account_value = psp_alloc*(1+psp_r.iloc[step]) + ghp_alloc*(1+ghp_r.iloc[step])
        peak_value = np.maximum(peak_value, account_value)
        w_history.iloc[step] = psp_w
    return w_history

def terminal_values(rets):
    """
    Returns the final values of a dollar at the end of the return period for each scenario
    """
    return (rets+1).prod()

def terminal_stats(rets, floor=.8, cap=np.inf, name='Stats'):
    terminal_wealth = (rets+1).prod()
    breach = terminal_wealth < floor
    reach = terminal_wealth >= cap
    p_breach = breach.mean() if breach.sum() > 0 else np.nan
    p_reach = breach.mean() if reach.sum() > 0 else np.nan
    e_short = (floor-terminal_wealth[breach]).mean() if breach.sum() > 0 else np.nan
    e_surplus = (cap-terminal_wealth[reach]).mean() if reach.sum() > 0 else np.nan
    sum_stats = pd.DataFrame.from_dict({
        'mean': terminal_wealth.mean(),
        'std': terminal_wealth.std(),
        'p_breach': p_breach,
        'e_short': e_short,
        'p_reach': p_reach,
        'e_surplus': e_surplus
    }, orient='index', columns=[name])
    return sum_stats

"""
********** FACTOR MODELS **********
"""
def tracking_error(r_a, r_b):
    """
    Returns the Tracking Error between two return series
    """
    return np.sqrt(((r_a - r_b)**2).sum())

def portfolio_tracking_error(weights, ref_r, bb_r):
    """
    Returns the tracking error between the reference returns
    and a portfolio of building block returns held with given weights.
    """
    return tracking_error(ref_r, (weights*bb_r).sum(axis=1))

def sharpe_style_analysis(dependent_variable, explanatory_variables):
    """
    Returns the optimal weights that minimize Tracking error between a portfolio
    of the explanatory variables and the dependent variable
    """
    n = explanatory_variables.shape[1]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0), ) * n # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
                        }
    solution = minimize(portfolio_tracking_error, init_guess,
                        args=(dependent_variable, explanatory_variables,), method='SLSQP',
                        options={'disp': False},
                        constraints=(weights_sum_to_1,),
                        bounds=bounds)
    weights = pd.Series(solution.x, index=explanatory_variables.columns)
    return weights

def backtest_ws(r, estimation_window=60, weighting=weight_ew, verbose=False, **kwargs):
    """
    Backtests a given weighting scheme, given some parameters:
    r : asset returns to use to build the portfolio
    estimation_window: the window to use to estimate parameters
    weighting: the weighting scheme to use, must be a function that takes "r", and a variable number of keyword-value arguments
    """
    n_periods = r.shape[0]
    # return windows
    windows = [(start, start+estimation_window) for start in range(n_periods-estimation_window)]
    weights = [weighting(r.iloc[win[0]:win[1]], **kwargs) for win in windows]
    # convert List of weights to DataFrame
    weights = pd.DataFrame(weights, index=r.iloc[estimation_window:].index, columns=r.columns)
    returns = (weights * r).sum(axis="columns",  min_count=1) #mincount is to generate NAs if all inputs are NAs
    return returns
    
"""
********** EXTRACTING EXPECTED RETURNS **********
"""
# Black Litterman implementation
def as_colvec(x):
    """
    Takes a numpy array or a numpy one-column matrix (a vector)
    and returns the data as a column vector.
    """
    if (x.ndim == 2):
        return x
    else:
        return np.expand_dims(x, axis=1)
    
def implied_returns(delta, sigma, w):
    """
    Obtain the implied expected returns by reverse engineering the weights
    Inputs:
    delta: Risk Aversion Coefficient (scalar)
    sigma: Variance-Covariance Matrix (N x N) as DataFrame
        w: Portfolio weights (N x 1) as Series
    Returns an N x 1 vector of Returns as Series
    """
    ir = delta * sigma.dot(w).squeeze() # to get a series from a 1-column dataframe
    ir.name = 'Implied Returns'
    return ir

def proportional_prior(sigma, tau, p):
    """
    Returns the He-Litterman simplified Omega
    Inputs:
    sigma: N x N Covariance Matrix as DataFrame
    tau: a scalar
    p: a K x N DataFrame linking Q and Assets
    returns a P x P DataFrame, a Matrix representing Prior Uncertainties
    """
    helit_omega = p.dot(tau * sigma).dot(p.T)
    # Make a diag matrix from the diag elements of Omega
    return pd.DataFrame(np.diag(np.diag(helit_omega.values)),index=p.index, columns=p.index)

def bl(w_prior, sigma_prior, p, q,
    omega=None,
    delta=2.5, tau=.02):
    """
    # Computes the posterior expected returns based on 
    # the original black litterman reference model
    #
    # W.prior must be an N x 1 vector of weights, a Series
    # Sigma.prior is an N x N covariance matrix, a DataFrame
    # P must be a K x N matrix linking Q and the Assets, a DataFrame
    # Q must be an K x 1 vector of views, a Series
    # Omega must be a K x K matrix a DataFrame, or None
    # if Omega is None, we assume it is
    #    proportional to variance of the prior
    # delta and tau are scalars
    """
    if omega is None:
        omega = proportional_prior(sigma_prior, tau, p)
    # Force w.prior and Q to be column vectors
    # How many assets do we have?
    N = w_prior.shape[0]
    # And how many views?
    K = q.shape[0]
    # First, reverse-engineer the weights to get pi
    pi = implied_returns(delta, sigma_prior,  w_prior)
    # Adjust (scale) Sigma by the uncertainty scaling factor
    sigma_prior_scaled = tau * sigma_prior  
    # posterior estimate of the mean, use the "Master Formula"
    # we use the versions that do not require
    # Omega to be inverted (see previous section)
    # this is easier to read if we use '@' for matrixmult instead of .dot()
    #     mu_bl = pi + sigma_prior_scaled @ p.T @ inv(p @ sigma_prior_scaled @ p.T + omega) @ (q - p @ pi)
    mu_bl = pi + sigma_prior_scaled.dot(p.T).dot(inv(p.dot(sigma_prior_scaled).dot(p.T) + omega).dot(q - p.dot(pi).values))
    # posterior estimate of uncertainty of mu.bl
#     sigma_bl = sigma_prior + sigma_prior_scaled - sigma_prior_scaled @ p.T @ inv(p @ sigma_prior_scaled @ p.T + omega) @ p @ sigma_prior_scaled
    sigma_bl = sigma_prior + sigma_prior_scaled - sigma_prior_scaled.dot(p.T).dot(inv(p.dot(sigma_prior_scaled).dot(p.T) + omega)).dot(p).dot(sigma_prior_scaled)
    return (mu_bl, sigma_bl)

def inverse(d):
    """
    Invert the dataframe by inverting the underlying matrix
    """
    return pd.DataFrame(inv(d.values), index=d.columns, columns=d.index)

def w_msr(sigma, mu, scale=True):
    """
    Optimal (Tangent/Max Sharpe Ratio) Portfolio weights
    by using the Markowitz Optimization Procedure
    Mu is the vector of Excess expected Returns
    Sigma must be an N x N matrix as a DataFrame and Mu a column vector as a Series
    This implements page 188 Equation 5.2.28 of
    "The econometrics of financial markets" Campbell, Lo and Mackinlay.
    """
    w = inverse(sigma).dot(mu)
    if scale:
        w = w/sum(w) # fix: this assumes all w is +ve
    return w

def w_star(delta, sigma, mu):
    return (inverse(sigma).dot(mu))/delta

"""
********** RISK CONTRIBUTION & RISK PARITY **********
"""
def risk_contribution(w, cov):
    """
    Compute the contributions to risk of the constituents of a portfolio,
    given a set of portfolio weights and a covariance matrix.
    """
    total_portfolio_var = portfolio_vol(w, cov)**2
    # Marginal contribution of each constituent
    marginal_contrib = cov@w
    risk_contrib = np.multiply(marginal_contrib, w.T)/total_portfolio_var
    return risk_contrib

def target_risk_contributions(target_risk, cov):
    """
    Returns the weights of the portfolio that gives you the weights such
    that the contributions to portfolio risk are as close as possible to
    the target_risk, given the covariance matrix.
    """
    n = cov.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n #an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    def msd_risk(weights, target_risk, cov):
        """
        Returns the Mean Squared Differenced in risk contributions between weights and target_risk.
        """
        w_contribs = risk_contribution(weights, cov)
        return ((w_contribs-target_risk)**2).sum()
    
    weights = minimize(msd_risk, init_guess,
                       args=(target_risk, cov),
                       method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,),
                       bounds=bounds
                       )
    return weights.x

def equal_risk_contributions(cov):
    """
    Returns the weights of the portfolio that equalizes the contributions
    of the constituents based on the given covariance matrix.
    """
    n = cov.shape[0]
    return target_risk_contributions(target_risk=np.repeat(1/n, n), cov=cov)

def weight_erc(r, cov_estimator=sample_cov, **kwargs):
    """
    Produces the weights of the ERC portfolio given a covariance matrix of the returns
    """
    est_cov = cov_estimator(r, **kwargs)
    return equal_risk_contributions(est_cov)

"""
********** TIME SERIES ANALYSIS **********
"""
def corr_vars_vs_lags(time_series_df, n_lags=20):
    """
    Function to calculate the correlation between each variable in a dataframe
    and lags from 0 to n_lags of all variables in the dataframe.\n
    Returns a dictionary with a dataframe of correlations for each variable.
    """
    correlation_dataframes = {}
    for main_var in time_series_df.columns:
        corrs = pd.DataFrame(columns=time_series_df.columns.values, index=range(n_lags))
        for var in corrs.columns:
            correlations = []
            for i in range(n_lags):
                corr_value = np.corrcoef(time_series_df[main_var].values[i:], time_series_df[var].shift(i).values[i:])[0, 1]
                correlations.append(corr_value)
            corrs[var] = correlations
        correlation_dataframes[main_var] = corrs
    return correlation_dataframes

def invert_transformation(first_val, df_forecast, log_transformation=False):
    """
    Invert the transformation of calculating the first (or second) difference of a time series.
    If first difference was calculated to transform a time series into a stationary series,
    this function returns the original values by providing:\n
    last_val: a series with the original first value for each variable in the time series dataframe.
    """
    for col in df_forecast.columns:
        df_forecast[col] = first_val[col] + df_forecast[col].cumsum()
    if log_transformation == True:
        df_forecast = np.exp(df_forecast)
    return df_forecast

"""
********** RISK MODELING **********
"""
# Weight of Evidence
def weight_of_evidence(data, x, y):
    """
    Return the weight of evidence for each category of an independent variable.\n
    x and y are the column names for the independent and dependent variable, respectively.\n
    x can be a variable name or a list of variable names.\n
    y must be a binary outcome.\n\n
    WoE_i = ln(%(y=1)_i / %(y=0)_i)\n
    Values range from about -4.6 to about 4.6.\n
    The farther away from 0, the more the independent variable separates outcomes .
    """
    n_success, n_failure = data[y].value_counts().get([1, 0], [0, 0])

    if isinstance(x, list):
        dummies_list = []
        for var in x:
            category_dummies = pd.get_dummies(data[var], prefix=var, prefix_sep=':')
            dummies_list.append(category_dummies)
        category_dummies = pd.concat(dummies_list, axis=1)
    else:
        category_dummies = pd.get_dummies(data[x], prefix=x, prefix_sep=':')
    outcome_by_category = pd.concat([data[y], category_dummies], axis=1).groupby(y).sum().T

    success_proportion = outcome_by_category[1]/n_success
    failure_proportion = outcome_by_category[0]/n_failure

    woe = np.log(success_proportion/failure_proportion)
    return woe

# Information value components
def information_value_components(data, x, y):
    """
    Return the information value of each category of an independent variable or list of variables.\n
    x and y are the column names for the independent and dependent variable, respectively.\n
    x can be a variable name or a list of variable names.\n
    y must be a binary outcome.\n\n
    IV = sum((%(y=1) - %(y=0))*WoE)\n
    IV = sum((%(y=1) - %(y=0))*ln(%(y=1)/%(y=0)))\n\n
    Information value ranges from 0 - 1, where typically:\n
    IV < 0.02: No predictive power\n
    0.02 < IV < 0.1: Weak predictive power\n
    0.1 < iv < 0.3: Medium predictive power\n
    0.3 < IV < 0.5: Strong predictive power\n
    0.5 < IV: Duspisciously high predictive power, too good to be true
    """
    n_success, n_failure = data[y].value_counts().get([1, 0], [0, 0])

    if isinstance(x, list):
        dummies_list = []
        for var in x:
            category_dummies = pd.get_dummies(data[var], prefix=var, prefix_sep=':')
            dummies_list.append(category_dummies)
        category_dummies = pd.concat(dummies_list, axis=1)
    else:
        category_dummies = pd.get_dummies(data[x], prefix=x, prefix_sep=':')
    outcome_by_category = pd.concat([data[y], category_dummies], axis=1).groupby(y).sum().T

    success_proportion = outcome_by_category[1]/n_success
    failure_proportion = outcome_by_category[0]/n_failure

    success_failue_diff = success_proportion - failure_proportion
    woe = np.log(success_proportion/failure_proportion)
    return success_failue_diff*woe

# Information value
def information_value(data, x, y):
    """
    Return the information value of an independent variable or a series of independent variables.\n
    x and y are the column names for the independent and dependent variable, respectively.\n
    x can be a variable name or a list of variable names.\n
    y must be a binary outcome.\n\n
    IV = sum((%(y=1) - %(y=0))*WoE)\n
    IV = sum((%(y=1) - %(y=0))*ln(%(y=1)/%(y=0)))\n\n
    Information value ranges from 0 - 1, where typically:\n
    IV < 0.02: No predictive power\n
    0.02 < IV < 0.1: Weak predictive power\n
    0.1 < iv < 0.3: Medium predictive power\n
    0.3 < IV < 0.5: Strong predictive power\n
    0.5 < IV: Duspisciously high predictive power, too good to be true
    """
    n_success, n_failure = data[y].value_counts().get([1, 0], [0, 0])

    if not isinstance(x, list):
        x = [x]

    iv_dataframe = pd.DataFrame(
        index=x,
        columns=['Information value']
    )

    for var in x:
        category_dummies = pd.get_dummies(data[var], prefix=var, prefix_sep=':')
        outcome_by_category = pd.concat([data[y], category_dummies], axis=1).groupby(y).sum().T

        success_proportion = outcome_by_category[1]/n_success
        failure_proportion = outcome_by_category[0]/n_failure

        woe = np.log(success_proportion/failure_proportion)
        iv = ((success_proportion - failure_proportion)*woe).sum()
        iv_dataframe.loc[var, 'Information value'] = iv

    return iv_dataframe

def woe_table(data, x, y):
    """
    Generate a table with details for the calculation of Weight of Evidence and Information Value
    from an independent variable explaining an outcome discrete variable.\n
    x and y are the column names for the independent and dependent variable, respectively.\n
    x can be a variable name or a list of variable names.\n
    y must be a binary outcome.\n\n
    """
    if not isinstance(x, list):
        x = [x]

    iv_detail_dfs = []
    for var in x:
        df = data[[var, y]]

        # Calculate counts by grade and proportion of good_bad lenders
        df = pd.concat([
            df.groupby(df.columns.values[0], as_index=False)[df.columns.values[1]].count(),
            df.groupby(df.columns.values[0], as_index=False)[df.columns.values[1]].mean()
        ], axis=1)
        df['variable'] = var

        df = df.iloc[:, [4, 0, 1, 3]]
        df.columns = ['variable', 'category', 'n_obs', 'prop_good']

        # Calculate proportions of successes and failures for each grade category
        df['prop_n_obs'] = df['n_obs']/df['n_obs'].sum()
        df['n_good'] = df['n_obs']*df['prop_good']
        df['n_bad'] = df['n_obs']*(1 - df['prop_good'])
        df['prop_n_good'] = df['n_good']/df['n_good'].sum()
        df['prop_n_bad'] = df['n_bad']/df['n_bad'].sum()

        # Calculate the WoE (Weight of Evidence)
        df['WoE'] = np.log(df['prop_n_good']/df['prop_n_bad'])
        if pd.api.types.is_object_dtype(df['category']):
            df.sort_values(by='WoE', ascending=True, inplace=True)
            df.reset_index(drop=True, inplace=True)
        df['diff_prop_good'] = df['prop_good'].diff().abs()
        df['diff_WoE'] = df['WoE'].diff().abs()

        df['IV'] = (df['prop_n_good'] - df['prop_n_bad']) * df['WoE']
        df['IV'] = df['IV'].sum()
        iv_detail_dfs.append(df)
    
    return pd.concat(iv_detail_dfs, axis=0)

def plot_by_woe(data, x, y, rotation_of_x_axis_labels=0):
    """
    Generate a plot to visualize the Weight of Evidence and Information Value
    from an independent variable explaining an outcome discrete variable.\n
    x and y are the column names for the independent and dependent variable, respectively.\n
    x can be a variable name or a list of variable names.\n
    y must be a binary outcome.\n\n
    """
    df_WoE = woe_table(data, x, y)
    variables = list(df_WoE['variable'].unique())

    if len(variables) > 1:
        fig, axes = plt.subplots(len(variables), 1, figsize=(18, 4*len(variables)))
        for var in variables:
            df_t = df_WoE[df_WoE['variable'] == var]
            x = np.array(df_t['category'].astype(str))
            y = df_t['WoE']
            ax = axes[variables.index(var)]

            ax.plot(x, y, marker='o', linestyle='--', color='k')
            ax.set_xlabel(var)
            ax.set_ylabel('Weight of Evidence')
            ax.set_title(str('Weight of Evidence by ' + var))
            ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=rotation_of_x_axis_labels)
            ax.grid(alpha=.3)
        plt.tight_layout()
    else:
        x = np.array(df_WoE['category'].astype(str))
        y = df_WoE['WoE']
        plt.figure(figsize=(18, 6))
        plt.plot(x, y, marker='o', linestyle='--', color='k')
        plt.xlabel(variables[0])
        plt.ylabel('Weight of Evidence')
        plt.title(str('Weight of Evidence by ' + variables[0]))
        plt.xticks(rotation=rotation_of_x_axis_labels)
        plt.grid(alpha=.3)

"""
********** TECHNICAL INDICATORS **********
"""
def technical_indicators(series, indicators=['SMA', 'EMA', 'MACD', 'SO', 'PRC'], ma_terms=10, macd_params=[12, 26, 9], so_params=[14, 3], plot=True, return_df=True, periods_to_plot=0, signal_tolerance=1):
    """
    Function that calculates and plots technical indicators included in the <indicators> list object for the given series.\n
    <series> must be a pandas series.\n
    <ma_terms> indicates the size of the time window for each of the indicators.\n
    <periods_to_plot> can limit the number of periods plotted by the given number.\n
    <indicators> list object can include any of the following indicators:\n
    <signal_tolerance> is a multiplier to the indicator value to determine the signal threshold.\n
    -SMA: Simple Moving Average\n
    -EMA: Exponential Moving Average\n
    -MACD: Moving Average Convergence Divergence\n
    -SO: Stochastic Oscilator\n
    -PRC: Average price over the most recent 20% of observations relative to 10% of the series maximum price
    """
    ma_terms = [ma_terms] if type(ma_terms) != list else ma_terms
    indicators_list = [series]
    signals = []
    n_extra_plots = 0
    
    # Computation of Simple Moving Averages
    if 'SMA' in indicators:
        for i in ma_terms:
            SMA = series.rolling(i).mean()
            SMA.name = 'SMA'+str(i)
            indicators_list.append(SMA)

            SMA_signals = series > SMA*signal_tolerance
            SMA_signals.name = SMA.name + ' Signal'
            signals.append(SMA_signals)

    # Computation of Exponential Moving Averages
    if 'EMA' in indicators:
        for i in ma_terms:
            EMA = series.ewm(span=i, adjust=False).mean()
            EMA.name = 'EMA'+str(i)
            indicators_list.append(EMA)

            EMA_signals = series > EMA*signal_tolerance
            EMA_signals.name = EMA.name + ' Signal'
            signals.append(EMA_signals)

    # Computation of Moving Average Convergence Divergence
    if 'MACD' in indicators:
        n_extra_plots += 1
        EMA_short_term = series.ewm(span=macd_params[0], adjust=False).mean()
        EMA_long_term = series.ewm(span=macd_params[1], adjust=False).mean()
        MACD = EMA_short_term - EMA_long_term
        signal_line = MACD.ewm(span=macd_params[2], adjust=False).mean()
        macd_series = pd.concat([MACD, signal_line], axis=1)
        macd_series.columns = [f'MACD ({macd_params[0]}, {macd_params[1]})', f'Signal line ({macd_params[2]})']

        MACD_signals = MACD > signal_line*signal_tolerance
        MACD_signals.name = 'MACD Signal'
        signals.append(MACD_signals)

    if 'SO' in indicators:
        n_extra_plots += 1
        SO_high = series.rolling(so_params[0]).max()
        SO_low = series.rolling(so_params[0]).min()
        pct_K = (series - SO_low)/(SO_high - SO_low)
        pct_D = pct_K.rolling(so_params[1]).mean()
        SO_series = pd.concat([pct_K, pct_D], axis=1)
        SO_series.columns = ['%K', '%D']

        SO_signals = pct_K > pct_D*signal_tolerance
        SO_signals.name = 'Stochastic Oscillator Signal'
        signals.append(SO_signals)
    
    if 'PRC' in indicators:
        n_periods = len(series)
        last_20_pct_count = max(1, int(n_periods * 0.2))
        avg_last_20_pct = series.tail(last_20_pct_count).mean()
        max_value_threshold = 0.1 * series.max()
        
        PRC_signals = pd.Series(avg_last_20_pct > max_value_threshold, index=series.index, name='PRC Signal')
        signals.append(PRC_signals)

    technical_indicators_df = pd.concat(indicators_list, axis=1)
    signals_df = pd.concat(signals, axis=1)

    if plot:
        ax_n = 1
        height_ratios = [2] + [1 for i in range(n_extra_plots)]
        fig, axes = plt.subplots(1+n_extra_plots, 1, figsize=(12, 5+n_extra_plots*3), height_ratios=height_ratios)

        if 'MACD' in indicators:
            macd_series.iloc[-periods_to_plot:].plot(ax=axes[ax_n])
            axes[ax_n].grid(alpha=.4)
            ax_n += 1
        if 'SO' in indicators:
            SO_series.iloc[-periods_to_plot:].plot(ax=axes[ax_n])
            axes[ax_n].axhline(.8, color='r')
            axes[ax_n].axhline(.2, color='r')
            axes[ax_n].grid(alpha=.4)
            ax_n += 1
        technical_indicators_df.iloc[-periods_to_plot:].plot(ax=axes[0])
        axes[0].grid(alpha=.4)
        axes[0].set_title('Technical indicators for ' + series.name)
        plt.tight_layout()
    if return_df:
        return pd.concat([technical_indicators_df, signals_df], axis=1)