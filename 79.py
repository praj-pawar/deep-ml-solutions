import math


def binomial_probability(n, k, p):
    """
    Calculate the probability of achieving exactly k successes in n independent Bernoulli trials,
    each with probability p of success, using the Binomial distribution formula.
    """
    # Compute n choose k
    nCk = math.comb(n, k)

    # Calculate probability using the Binomial formula
    probability = nCk * (p ** k) * ((1 - p) ** (n - k))

    return round(probability, 5)
