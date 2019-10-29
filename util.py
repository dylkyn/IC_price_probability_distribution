import requests

import numpy as np
import pandas as pd


def get_option_data(ticker: str, expiration_date: str) -> (float, pd.DataFrame):
    """
    :param ticker: stock ticker
    :param expiration_date: option chain expiration date (YYYY-MM-DD)
    :return: tuple containing stock price and data frame with option chain
    """
    stock_price_response = requests.get(
        'https://sandbox.tradier.com/v1/markets/quotes',
        params={
            'symbols': ticker,
        },
        headers={
            'Authorization': 'Bearer RudqnB0bzRxQVucDkn3dKTx1Ka34',
            'Accept': 'application/json'
        }
    )
    try:
        stock_price = stock_price_response.json()['quotes']['quote']['last']
    except KeyError:
        raise ValueError(f"Ticker {ticker} is not valid")

    option_price_response = requests.get(
        'https://sandbox.tradier.com/v1/markets/options/chains',
        params={
            'symbol': ticker,
            'expiration': expiration_date,
        },
        headers={
            'Authorization': 'Bearer RudqnB0bzRxQVucDkn3dKTx1Ka34',
            'Accept': 'application/json'
        }
    )
    try:
        options = option_price_response.json()['options']['option']
    except TypeError:
        raise ValueError("Invalid expiration date")

    data = []
    for i in range(0, len(options), 2):
        strike_price = options[i]['strike']
        option_type = options[i]['option_type']
        if option_type == 'call':
            call_price = options[i]['last']
            put_price = options[i + 1]['last']
        else:
            call_price = options[i + 1]['last']
            put_price = options[i]['last']
        data.append([strike_price, call_price, put_price])
    df = pd.DataFrame(data, columns=['strike_price', 'call_price', 'put_price'])
    return stock_price, df


def get_dummy_option_data() -> pd.DataFrame:
    """
    Returns a pandas data frame with the AAPL option chain expiring 10/11/19.
    At the time of creation (10/06/19), AAPL price is 227.01.
    col1 = strike price
    col2 = call price
    col3 = put price
    """
    contracts = {'strike_price': [207.5 + 2.5 * x for x in range(17)],
                 'call_price': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                                2.82, 1.7, .92, .47, .25, .15, .11, .06, .06],
                 'put_price': [.2, .27, .37, .5, .7, 1.01, 1.53, 2.19, 3.25,
                               np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]}
    df = pd.DataFrame(contracts, columns=['strike_price', 'call_price', 'put_price'])
    return df
