import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import util


def build_model(option_chain: pd.DataFrame, debug=False) -> np.ndarray:
    # PUTS
    puts = option_chain[['strike_price', 'put_price']].dropna()
    p_below = 0.02  # Guess for probability of stock being below lowest strike put
    put_probabilities = [p_below]
    i = 0
    while i < puts.shape[0] - 2:
        p_between = calc_put_prob(puts=puts, p_below=p_below, short_index=i, long_index=i+2, debug=debug)
        p_below += p_between
        put_probabilities.append(p_between)
        i += 2

    if debug:
        print('-------------------------------------------------')

    # CALLS
    calls = option_chain[['strike_price', 'call_price']].dropna()
    p_above = 0.001  # Guess for probability of stock being above highest strike call
    call_probabilities = [p_above]
    # Start at the bottom of the array since we guess probability above instead of using previously calculated
    # p_below to reduce effect of initial p_below guess error
    i = calls.shape[0] - 1
    while i > 0:
        p_between = calc_call_prob(calls=calls, p_above=p_above, long_index=i-2, short_index=i, debug=debug)
        p_above += p_between
        call_probabilities.append(p_above)
        i -= 2
    call_probabilities.reverse()
    return np.array(put_probabilities + call_probabilities) * 100


def calc_put_prob(puts: pd.DataFrame, p_below: float, short_index: int, long_index: int, debug=False) -> float:
    short_put = puts.iloc[short_index]
    long_put = puts.iloc[long_index]

    max_loss = short_put['put_price'] - long_put['put_price']  # max_loss is negative
    max_gain = long_put['strike_price'] - short_put['strike_price'] + max_loss
    avg_gain = (max_gain + max_loss) / 2

    # Solve system of equations
    a = np.array([[avg_gain, max_loss], [1.0, 1.0]])
    b = np.array([[0 - max_gain * p_below], [1 - p_below]])
    x = np.linalg.solve(a, b)

    p_between = x[0][0]
    p_above = x[1][0]

    if debug:
        print('#################')
        print("Probability below ${} is {}%".format(short_put['strike_price'], p_below * 100))
        print("Probability between ${} and ${} is {}%".format(short_put['strike_price'], long_put['strike_price'],
                                                              p_between * 100))
        print("Probability above ${} is {}%".format(long_put['strike_price'], p_above * 100))

    return p_between


def calc_call_prob(calls: pd.DataFrame, p_above: float, long_index: int, short_index: int, debug=False) -> float:
    long_call = calls.iloc[long_index]
    short_call = calls.iloc[short_index]

    max_loss = short_call['call_price'] - long_call['call_price']  # max_loss is negative
    max_gain = short_call['strike_price'] - long_call['strike_price'] + max_loss
    avg_gain = (max_gain + max_loss) / 2

    # Solve system of equations
    a = np.array([[max_loss, avg_gain], [1.0, 1.0]])
    b = np.array([[0 - max_gain * p_above], [1 - p_above]])
    x = np.linalg.solve(a, b)

    p_below = x[0][0]
    p_between = x[1][0]

    if debug:
        print('#################')
        print("Probability above ${} is {}%".format(short_call['strike_price'], p_above * 100))
        print("Probability between ${} and ${} is {}%".format(long_call['strike_price'], short_call['strike_price'],
                                                              p_between * 100))
        print("Probability below ${} is {}%".format(long_call['strike_price'], p_below * 100))

    return p_between


def plot_pd(option_chain: pd.DataFrame, probs: np.ndarray):
    bins = option_chain['strike_price'].to_numpy()[::2]
    x = bins[:-1] + 1
    weights = probs[1:-1]  # Exclude edge probabilities since these do not fit in histogram
    n, bins, patches = plt.hist(x=x, bins=bins, weights=weights, edgecolor='k')
    plt.xticks(bins)
    plt.xlabel('Strike Price ($)')
    plt.ylabel('Probability')
    plt.show()
    pass


if __name__ == "__main__":
    option_chain = util.get_dummy_option_data()
    probabilities = build_model(option_chain, debug=False)
    plot_pd(option_chain, probabilities)

