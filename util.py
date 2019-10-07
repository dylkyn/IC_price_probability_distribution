import numpy as np


def get_dummy_option_data() -> (float, np.ndarray):
    """
    Returns a tuple containing the APPLE current price (10/06/19)
    and a numpy array with the option chain expiring 10/11/19.
    col1 = strike price
    col2 = call price
    col3 = put price
    """
    current_price = 227.01
    array = np.zeros((18, 3))
    array[:, 0] = [205 + 2.5*x for x in range(18)]
    array[:, 1] = [22.26, 19.7, 17.18, 14.97, 12.66, 10.3, 7.94, 6, 4.3, 2.82, 1.7, .92, .47, .25, .15, .11, .06, .06]
    array[:, 2] = [.16, .2, .27, .37, .5, .7, 1.01, 1.53, 2.19, 3.25, 4.55, 6.4, 8.35, 10.8, 13.6, 16.15, 17.9, 0]
    return current_price, array
