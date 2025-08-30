import math


def fact(n):
    if n == 0:
        return 1
    return n*fact(n-1)


def poisson_probability(k, lam):
    """
    Calculate the probability of observing exactly k events in a fixed interval,
    given the mean rate of events lam, using the Poisson distribution formula.
    :param k: Number of events (non-negative integer)
    :param lam: The average rate (mean) of occurrences in a fixed interval
    """
    # Your code here
    value = (math.exp(-lam)*pow(lam, k))/fact(k)
    return round(value, 5)
