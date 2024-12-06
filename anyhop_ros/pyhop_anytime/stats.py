import time
from typing import *
import numpy as np


def report_one(label: str, plan_times) -> float:
    print(label)
    print(f"{len(plan_times)} plans")
    if len(plan_times) > 0:
        print(f"steps: {plan_times[-1][1]:.2f} ({plan_times[-1][2]:.2f}s)")
        print()
        return plan_times[-1][1]


def get_confidence_interval(values: List[float]) -> Tuple[float, float, float]:
    # From https://scales.arabpsychology.com/stats/calculate-confidence-intervals-in-python/
    mean = np.mean(values)
    if len(values) < 30:
        lo, hi = t_interval(0.95, df=len(values) - 1, loc=mean, scale=sem(values))
    else:
        lo, hi = norm_interval(0.95, df=len(values) - 1, loc=mean, scale=sem(values))
    return lo, mean, hi


def report_confidence_interval(values: List[float]):
    lo, mean, hi = get_confidence_interval(values)
    return f"95% confidence interval: ({lo:.2f}, {mean:.2f}, {hi:.2f})"


def experiment(num_problems: int, runs_per_problem: int, max_seconds: float, problem_generator,
               non_random_planners: Dict,
               random_planners: Dict) -> Dict[str, np.ndarray]:
    result = {}
    for name in list(non_random_planners) + list(random_planners):
        result[name] = np.zeros((num_problems, 2))
    start_time = time.time()
    for i in range(num_problems):
        print(f"Problem {i + 1}")
        state, tasks = problem_generator()
        non_random_costs = {}
        for name, planner in non_random_planners.items():
            plan_times = planner(state, tasks, max_seconds)
            cost = report_one(name, plan_times)
            if cost is not None:
                non_random_costs[name] = cost

        random_costs = {}
        for name, planner in random_planners.items():
            random_costs[name] = []
            for j in range(runs_per_problem):
                print(f"Run {j + 1}")
                plan_times = planner(state, tasks, max_seconds)
                cost = report_one(name, plan_times)
                if cost is not None:
                    random_costs[name].append(cost)

        print(f"Problem {i + 1} summary")
        longest_name_len = max(len(name) for name in non_random_costs)
        longest_name_len = max(longest_name_len, max(len(name) for name in random_costs))
        for name, cost in non_random_costs.items():
            print(f"{name}:{' ' * (longest_name_len - len(name))}{cost}")
            result[name][i][0] = cost
            result[name][i][1] = 0
        for name, costs in random_costs.items():
            print(f"{name}:{' ' * (longest_name_len - len(name))}{report_confidence_interval(costs)}")
            lo, mean, hi = get_confidence_interval(costs)
            result[name][i][0] = mean
            result[name][i][1] = hi - mean

        print()
        print()
    duration = time.time() - start_time
    print(f"Duration: {duration:.2f}s")
    return result


# Code from perplexity.ai, to replace scipy functions due to incompatibility with pypy
# https://www.perplexity.ai/search/rewrite-the-scipy-functions-t-dhuTCseDQuiwRwHvvchI0g#0


def t_interval(alpha, df, loc=0, scale=1):
    """
    Compute the confidence interval for the t-distribution.

    Parameters:
    - alpha: Confidence level (e.g., 0.95 for 95% confidence)
    - df: Degrees of freedom
    - loc: Location parameter (default: 0)
    - scale: Scale parameter (default: 1)

    Returns:
    - Tuple containing the lower and upper bounds of the confidence interval
    """
    p = alpha + (1 - alpha) / 2
    t_value = t_ppf(p, df)

    lower = loc - t_value * scale
    upper = loc + t_value * scale

    return (lower, upper)


def norm_interval(alpha, loc=0, scale=1):
    """
    Compute the confidence interval for the normal distribution.

    Parameters:
    - alpha: Confidence level (e.g., 0.95 for 95% confidence)
    - loc: Mean of the distribution (default: 0)
    - scale: Standard deviation of the distribution (default: 1)

    Returns:
    - Tuple containing the lower and upper bounds of the confidence interval
    """
    p = alpha + (1 - alpha) / 2
    z_value = norm_ppf(p)

    lower = loc - z_value * scale
    upper = loc + z_value * scale

    return (lower, upper)


def t_ppf(p, df):
    """
    Percent point function (inverse of cdf) for the t-distribution.
    This is a simplified approximation and may not be as accurate as SciPy's implementation.
    """
    if df == 1:
        return np.tan(np.pi * (p - 0.5))
    elif df == 2:
        return np.sign(p - 0.5) * np.sqrt(2 / (p * (1 - p)) - 2)
    else:
        # Use normal approximation for large df
        z = norm_ppf(p)
        y = z + (z ** 3 + z) / (4 * df) + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * df ** 2)
        return y * np.sqrt(df / (df - 2))


def norm_ppf(p):
    """
    Percent point function (inverse of cdf) for the standard normal distribution.
    This uses the Acklam's approximation.
    """
    a1 = -39.6968302866538
    a2 = 220.946098424521
    a3 = -275.928510446969
    a4 = 138.357751867269
    a5 = -30.6647980661472
    a6 = 2.50662827745924

    b1 = -54.4760987982241
    b2 = 161.585836858041
    b3 = -155.698979859887
    b4 = 66.8013118877197
    b5 = -13.2806815528857

    c1 = -7.78489400243029E-03
    c2 = -0.322396458041136
    c3 = -2.40075827716184
    c4 = -2.54973253934373
    c5 = 4.37466414146497
    c6 = 2.93816398269878

    if p <= 0 or p >= 1:
        raise ValueError("p must be between 0 and 1")

    if p < 0.5:
        q = p
    else:
        q = 1 - p

    r = np.sqrt(-2 * np.log(q))

    if r <= 5:
        r -= ((((c1 * r + c2) * r + c3) * r + c4) * r + c5) * r + c6
        r /= ((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6
    else:
        r -= ((((b1 * r + b2) * r + b3) * r + b4) * r + b5) / ((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6

    if p < 0.5:
        return -r
    else:
        return r


def sem(a, axis=0, ddof=1, nan_policy='propagate'):
    """
    Compute the standard error of the mean.

    Parameters:
    - a: Input array or object that can be converted to an array.
    - axis: Axis along which to calculate the standard error of the mean.
            If None, compute over the whole array. Default is 0.
    - ddof: Delta Degrees of Freedom. Default is 1.
    - nan_policy: {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains nan.
        The following options are available:
        'propagate': returns nan
        'raise': throws an error
        'omit': performs the calculations ignoring nan values

    Returns:
    - The standard error of the mean.
    """
    # Convert input to numpy array
    a = np.asarray(a)

    # Handle NaN values based on nan_policy
    if nan_policy not in ['propagate', 'raise', 'omit']:
        raise ValueError("nan_policy must be one of {'propagate', 'raise', 'omit'}")

    if nan_policy == 'raise':
        if np.isnan(a).any():
            raise ValueError("The input contains nan values")
    elif nan_policy == 'omit':
        a = a[~np.isnan(a)]

    # Calculate the standard deviation
    std = np.std(a, axis=axis, ddof=ddof)

    # If axis is None, n is the total number of elements
    # Otherwise, n is the size of the axis along which we're calculating
    if axis is None:
        n = a.size
    else:
        n = a.shape[axis]

    # Calculate and return the standard error of the mean
    return std / np.sqrt(n)