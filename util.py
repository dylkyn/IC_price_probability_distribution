import numpy as np
import pandas as pd


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
