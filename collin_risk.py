import pandas as pd
import numpy as np
import scipy
from scipy import stats
import yfinance as yf
from scipy.optimize import minimize


def var_historic(r, level = 5):
    """
    A function written by Collin Dougherty to calculate the historic VaR's of a dataframe.
    Returns the level = nth percentile of each column inputed in a dataframe.
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level = level)
    elif isinstance(r, pd.Series):
        return np.percentile(r, level)
    else:
        raise TypeError("Expected r to be a series or dataframe")
        

def var_cornish_fisher(r, level = 5):
    """
    Cornish-Fisher modified VaR.
    """
    z = scipy.stats.norm.ppf(level/100)
    s = scipy.stats.skew(r)
    k = scipy.stats.kurtosis(r)
    z = (z + 
        (z**2 - 1)*s/6 +
         (z**3 - 3*z)*(k-3)/24 -
         (2*z**3 - 5*z)*(s**2)/36
        )
    custom_var = r.mean() + z*r.std(ddof=0)
    traditional_var = 1 - custom_var
    return traditional_var


def get_stock_close_data(stocks = ['SPY', 'DIA', 'QQQ'], start_date = '2010-01-1', end_date = '2022-07-15', interval = '1d'):
    """
    Function to get closing price data for a list of stocks, input in form: stocks = ['SPY', 'DIA', 'QQQ'].
    Tickers must be yfinance tickers.
    """
    dates = pd.date_range(start_date, end_date)
    data = pd.DataFrame(index=dates)
    for i in stocks:
        data[i] = yf.download(tickers = i, start = start_date, end = end_date, interval = interval)[['Close']]
    clean_data = data.dropna()
    return clean_data    

    

def get_stock_return_data(stocks = ['SPY', 'DIA', 'QQQ'], start_date = '2010-01-1', end_date = '2022-07-15', interval = '1d'):
    """
    Function to get returns for a list of stocks, input in form: stocks = ['SPY', 'DIA', 'QQQ'].
    Tickers must be yfinance tickers.
    """
    dates = pd.date_range(start_date, end_date)
    data = pd.DataFrame(index=dates)
    returns = pd.DataFrame(index=dates)
    for i in stocks:
        data[i] = yf.download(tickers = i, start = start_date, end = end_date, interval = interval)[['Close']]
    data_clean = data.dropna()
    for i in stocks:
        returns[i] = (data_clean[i]-data_clean[i].shift(1))/data_clean[i].shift(1)
    returns_clean = returns.dropna()
    return returns_clean


def get_stock_cum_return_data(stocks = ['SPY', 'DIA', 'QQQ'], start_date = '2010-01-1', end_date = '2022-07-15', interval = '1d'):
    """
    Function to get returns for a list of stocks, input in form: stocks = ['SPY', 'DIA', 'QQQ'].
    Tickers must be yfinance tickers.
    """
    dates = pd.date_range(start_date, end_date)
    data = pd.DataFrame(index=dates)
    returns = pd.DataFrame(index=dates)
    for i in stocks:
        data[i] = yf.download(tickers = i, start = start_date, end = end_date, interval = interval)[['Close']]
    data_clean = data.dropna()
    for i in stocks:
        returns[i] = (((data_clean[i]-data_clean[i].shift(1))/data_clean[i].shift(1))+1).prod()
    cum_returns = returns.iloc[-1]
    adj_cum_returns = cum_returns - 1
    #table = pd.DataFrame({'Returns': adj_cum_returns})
    return adj_cum_returns


def annualized_data(daily_returns):
    """
    Expected input is a dataframe of daily returns for stocks.
    """
    adj_returns = daily_returns + 1
    prod_returns = adj_returns.cumprod()
    cum_returns = prod_returns.iloc[-1] - 1
    length = len(daily_returns.index)
    years = 252/length
    ann_returns = ((cum_returns+1)**years)-1
    ann_vol = daily_returns.std(ddof=0)*(252**0.5)
    annualized = pd.DataFrame({"Annualized Returns": ann_returns,
                               "Annualized Volatility": ann_vol})
    return annualized



def portfolio_return(weights, returns):
    """
    Weights to Returns of portfolio.
    """
    data = weights.T @ returns
    return data


def portfolio_vol(weights, covmat):
    """
    Weights to Vol of portfolio.
    """
    return (weights.T @ covmat @ weights)**0.5


def plot_efficient_frontier_2(stocks = ['SPY', 'QQQ'], n_points = 20, start_date = '2010-01-1', end_date = '2022-07-15', interval = '1d'):
    returns = get_stock_cum_return_data(stocks = stocks, start_date = start_date, end_date = end_date, interval = interval)-1
    covmat = get_stock_return_data(stocks = stocks, start_date = start_date, end_date = end_date, interval = interval).cov()
    n_points = n_points
    weights = [np.array([w, 1-w]) for w in np.linspace(0,1,n_points)]
    rets = [portfolio_return(w, returns) for w in weights]
    vols = [portfolio_vol(w, covmat) for w in weights]
    ef = pd.DataFrame({"R": rets,
                  "Vol": vols})
    return ef.plot.scatter(x='Vol', y='R', title = "Efficient Frontier")


def minimize_vol(cum_returns, covmat, target_return):
    """
    minimizes vol for a given return rate
    """
    #returns = get_stock_cum_return_data(stocks = stocks, start_date = start_date, end_date = end_date, interval = interval)-1
    #covmat = get_stock_return_data(stocks = stocks, start_date = start_date, end_date = end_date, interval = interval).cov()
    n = cum_returns.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),)*n
    return_is_target = {
        'type': 'eq',
        'args': (cum_returns,),
        'fun': lambda weights, cum_returns: target_return - portfolio_return(weights, cum_returns)
    }
    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) -1
    }
    results = minimize(portfolio_vol, init_guess,
                       args = (covmat,), method = "SLSQP",
                       options = {'disp': False},
                       constraints=(return_is_target, weights_sum_to_1),
                       bounds = bounds)
    return results.x


def optimal_weights(cum_returns, covmat, n_points):
    """
    generates weights
    """
    #returns = get_stock_cum_return_data(stocks = stocks, start_date = start_date, end_date = end_date, interval = interval)-1
    #covmat = get_stock_return_data(stocks = stocks, start_date = start_date, end_date = end_date, interval = interval).cov()
    target_rs = np.linspace(cum_returns.min(), cum_returns.max(), n_points)
    weights = [minimize_vol(cum_returns, covmat, target_return) for target_return in target_rs]
    return weights


def plot_ef(cum_returns, covmat, n_points = 30):
    """
    plots the efficient frontier
    """
    #returns = get_stock_return_data(stocks = stocks, start_date = start_date, end_date = end_date, interval = interval)-1
    #covmat = get_stock_return_data(stocks = stocks, start_date = start_date, end_date = end_date, interval = interval).cov()
    weights = optimal_weights(n_points = n_points, cum_returns = cum_returns, covmat = covmat)
    rets = [portfolio_return(w, cum_returns) for w in weights]
    vols = [portfolio_vol(w, covmat) for w in weights]
    ef = pd.DataFrame({"R": rets,
                  "Vol": vols})
    return ef.plot.scatter(x='Vol', y='R', title = "Efficient Frontier")
    
    
def maximize_sharpe_ratio(cum_returns, covmat, risk_free_rate):
    """
    risk free rate -> max sharpe
    """
    #returns = get_stock_cum_return_data(stocks = stocks, start_date = start_date, end_date = end_date, interval = interval)-1
    #covmat = get_stock_return_data(stocks = stocks, start_date = start_date, end_date = end_date, interval = interval).cov()
    n = cum_returns.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),)*n
    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) -1
    }
    def neg_sharpe_ratio(weights, risk_free_rate, cum_returns, covmat):
        """
        returns the negative of the sharpe ratio
        """
        r = portfolio_return(weights, cum_returns)
        vol = portfolio_vol(weights, covmat)
        return -(r-risk_free_rate)/vol
    results = minimize(neg_sharpe_ratio, init_guess,
                       args = (risk_free_rate, cum_returns, covmat,), method = "SLSQP",
                       options = {'disp': False},
                       constraints=(weights_sum_to_1),
                       bounds = bounds)
    return results.x


def max_sharpe_plot(ann_returns, covmat, risk_free_rate = 0.03, n_points = 30):
    """
    plots efficient frontier with max sharpe portfolio in red
    """
    plot = plot_ef(ann_returns, covmat, n_points = 30)
    #plot.set_xlim(left=0)
    risk_free_rate = risk_free_rate
    w_msr = maximize_sharpe_ratio(ann_returns, covmat, risk_free_rate)
    r_msr = portfolio_return(w_msr, ann_returns)
    vol_msr = portfolio_vol(w_msr, covmat)
    # Add CML (capital market line)
    #cml_x = [0, vol_msr]
    #cml_y = [risk_free_rate, r_msr]
    cml_x = [vol_msr]
    cml_y = [r_msr]
    return plot.plot(cml_x, cml_y, color = 'red', marker = 'o', linestyle = 'dashed')


def gmv(covmat):
    """
    weights of the asset allocation that minimizes volatility
    """
    n = covmat.shape[0]
    return maximize_sharpe_ratio(np.repeat(1,n), covmat, 0)


def plot_gmv(ann_returns, covmat):
    """
    plots the global minimum variance portfolio
    """
    plot = plot_ef(ann_returns, covmat, n_points = 30)
    w_gmv = gmv(covmat)
    r_gmv = portfolio_return(w_gmv, ann_returns)
    vol_gmv = portfolio_vol(w_gmv, covmat)
    gmv_x = [vol_gmv]
    gmv_y = [r_gmv]
    return plot.plot(gmv_x, gmv_y, color = 'goldenrod', marker = 'o', linestyle = 'dashed', markersize = 10)