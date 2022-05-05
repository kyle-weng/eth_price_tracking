from os import error
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import scipy.optimize as op
import MCMC, MCMC_error
import diagnostics
import yfinance

# disable chained assignments
pd.options.mode.chained_assignment = None 

yfinance.pdr_override()

eth = yfinance.download("ETH-USD", start="2017-11-09", end="2022-03-09")
# move date column from index to actual column
eth_date_column = eth.copy().reset_index()
eth_date_column["search_interest"] = eth_date_column["Close"].copy()
eth_date_column["std"] = eth_date_column["Close"].copy()

eth_search_interest = pd.read_csv("multiTimeline.csv")
eth_search_interest["Time"] = pd.to_datetime(eth_search_interest["Week"])

# manually impute
eth_date_column["search_interest"][0] = 10
eth_date_column["search_interest"][1] = 10
eth_date_column["search_interest"][2] = 10

# impute values
eth_dc_index = 3
eth_si_index = 0
while True:
    # print(eth_dc_index, eth_si_index)
    eth_dc_date = eth_date_column["Date"][eth_dc_index]
    eth_si_date = eth_search_interest["Time"][eth_si_index]
    if eth_dc_date >= eth_si_date:
        eth_date_column["search_interest"][eth_dc_index] = eth_search_interest["ethereum: (United States)"][eth_si_index]
        eth_dc_index += 1
    if eth_si_index < len(eth_search_interest) - 1 and eth_dc_date == eth_search_interest["Time"][eth_si_index + 1]:
        eth_si_index += 1
    if eth_dc_index == len(eth_date_column):
        break

# impute standard deviation
for i in range(1, len(eth_date_column) - 1):
    eth_date_column["std"][i] = np.std([eth_date_column["Close"][i + 1], eth_date_column["Close"][i], eth_date_column["Close"][i - 1]])

eth_date_column["std"][0] = np.std([eth_date_column["Close"][0], eth_date_column["Close"][1]])
eth_date_column["std"][len(eth_date_column) - 1] = np.std([eth_date_column["Close"][len(eth_date_column) - 1], eth_date_column["Close"][len(eth_date_column) - 2]])
# eth_date_column["std"] *= eth_date_column["Volume"] / np.mean(eth_date_column["Volume"])
# eth_date_column["std"] = (eth_date_column["High"] - eth_date_column["Low"]) / np.sqrt(eth_date_column["Volume"])


# transplant search_interest column back to original eth df
# without to_numpy(), we'll copy over NaNs because unalignable indices
eth["search_interest"] = eth_date_column["search_interest"].to_numpy()
eth["std"] = eth_date_column["std"].to_numpy()

# now we can plot them together
fig, ax1 = plt.subplots()
fig.set_figheight(5)
fig.set_figwidth(7.5)
plt.title("ETH-USD closing price and ETH Google search interest")
ax1.plot(eth["Close"], color='tab:red')
ax1.set_ylabel("Closing price (USD)", color='tab:red')
ax1.tick_params(axis='y', labelcolor='tab:red')
ax2 = ax1.twinx()
ax2.plot(eth["search_interest"], color='tab:blue')
ax2.set_ylabel("Google Trends search interest", color='tab:blue')
ax2.tick_params(axis='y', labelcolor='tab:blue')
ax1.set_xlabel("Date")
plt.savefig("ETH-USD closing price and ETH Google search interest.png")
plt.show()

# try multiple models
def get_linear_model(params: np.ndarray, x: float) -> float:
    assert len(params) == 2, "found {0} params instead".format(len(params))
    return params[0] + (params[1] * x)

def get_quadratic_model(params: np.ndarray, x: float) -> float:
    assert len(params) == 3, "found {0} params instead".format(len(params))
    return params[0] + (params[1] * x) + (params[2] * (x ** 2))

def get_cubic_model(params: np.ndarray, x: float) -> float:
    assert len(params) == 4, "found {0} params instead".format(len(params))
    return params[0] + (params[1] * x) + (params[2] * (x ** 2)) + (params[3] * (x ** 3))

def get_sigmoid_model(params: np.ndarray, x: float) -> float:
    assert len(params) == 4, "found {0} params instead".format(len(params))
    return params[0] / (1 + np.exp(-1 * params[2] * (x - params[1]))) + params[3]

interest = np.array(eth["search_interest"])
closing = np.array(eth["Close"])
err = np.array(eth["std"])
xs = np.linspace(np.min(interest), np.max(interest), 100)

# Log-likelihoods for each model fit - without error parameter
# equivalent to what the TA commented on my PS4, 1a
def negative_ln_likelihood(get_model, parameters: np.ndarray, x: float, y: float, err: float):
    first_term = np.log(1 / ((np.sqrt(2 * np.pi) * err)))
    second_term = (-1 / (2 * np.pi * np.power(err, 2)) * np.power(get_model(parameters, x) - y, 2))

    return -1 * np.sum(first_term + second_term)

# Log-likelihoods for each model fit - WITH error parameter
def negative_ln_likelihood_sys(get_model, parameters: np.ndarray, x: float, y: float):
    err = parameters[-1]
    parameters = parameters[:-1]

    first_term = np.log(1 / ((np.sqrt(2 * np.pi) * err)))
    second_term = (-1 / (2 * np.pi * np.power(err, 2)) * np.power(get_model(parameters, x) - y, 2))

    return -1 * np.sum(first_term + second_term)

# solely for use of minimizers
def negative_ln_likelihood_linear(parameters: np.ndarray, x: float, y: float, err: float):
    return negative_ln_likelihood(get_linear_model, parameters, x, y, err)

def negative_ln_likelihood_quad(parameters: np.ndarray, x: float, y: float, err: float):
    return negative_ln_likelihood(get_quadratic_model, parameters, x, y, err)

def negative_ln_likelihood_cubic(parameters: np.ndarray, x: float, y: float, err: float):
    return negative_ln_likelihood(get_cubic_model, parameters, x, y, err)

def negative_ln_likelihood_sigmoid(parameters: np.ndarray, x: float, y: float, err: float):
    return negative_ln_likelihood(get_sigmoid_model, parameters, x, y, err)

def negative_ln_likelihood_sys_linear(parameters: np.ndarray, x: float, y: float):
    return negative_ln_likelihood_sys(get_linear_model, parameters, x, y)

def negative_ln_likelihood_sys_quad(parameters: np.ndarray, x: float, y: float):
    return negative_ln_likelihood_sys(get_quadratic_model, parameters, x, y)

def negative_ln_likelihood_sys_cubic(parameters: np.ndarray, x: float, y: float):
    return negative_ln_likelihood_sys(get_cubic_model, parameters, x, y)

def negative_ln_likelihood_sys_sigmoid(parameters: np.ndarray, x: float, y: float):
    return negative_ln_likelihood_sys(get_sigmoid_model, parameters, x, y)

# linear (these parameter guesses are from the poly1d fits)----------------------------------------------------------
parameter_guess_linear = np.array([150, 18])
minimization_linear = op.minimize(negative_ln_likelihood_linear, parameter_guess_linear, args=(interest, closing, err))
# note that optimal estimation results match up with MCMC results

# check residuals
# plt.figure()
# plt.scatter(interest, closing - get_linear_model(minimization_linear.x, interest))
# plt.ylabel("Residual")
# plt.xlabel("Search interest")
# plt.title("Residuals - linear model without error parameter")
# plt.show()

# systematic error guess - std of residuals
parameter_guess_linear_sys = np.array([150, 18, np.std(closing - get_linear_model(minimization_linear.x, interest))])
minimization_linear_sys = op.minimize(negative_ln_likelihood_sys_linear, parameter_guess_linear_sys, args=(interest, closing))

# plt.figure()
# plt.scatter(interest, closing - get_linear_model(minimization_linear_sys.x[:-1], interest))
# plt.ylabel("Residual")
# plt.xlabel("Search interest")
# plt.title("Residuals - linear model with error parameter")
# plt.show()

# so the fit with the error parameter is significantly better

# can we try combining the two? (sigma_true^2 = sigma_measured^2 + sigma_systematic^2)
def negative_ln_likelihood_true(get_model, parameters: np.ndarray, x: float, y: float, err: float):
    # the last element of parameters is the sigma term

    systematic_error = parameters[-1]
    parameters = parameters[:-1]

    sigma_measured_squared = np.power(err, 2)
    sigma_systematic_squared = np.power(systematic_error, 2)
    sigma_true_squared = sigma_measured_squared + sigma_systematic_squared

    first_term = np.log(1 / ((np.sqrt(2 * np.pi) * np.sqrt(sigma_true_squared))))
    second_term = (-1 / (2 * np.pi * sigma_true_squared) * np.power(get_model(parameters, x) - y, 2))

    return -1 * np.sum(first_term + second_term)

def negative_ln_likelihood_true_linear(parameters: np.ndarray, x: float, y: float, err: float):
    return negative_ln_likelihood_true(get_linear_model, parameters, x, y, err)

minimization_linear_true = op.minimize(negative_ln_likelihood_true_linear, parameter_guess_linear_sys, args=(interest, closing, err))

# plt.figure()
# plt.scatter(interest, closing - get_linear_model(minimization_linear_true.x[:-1], interest))
# plt.ylabel("Residual")
# plt.xlabel("Search interest")
# plt.title("Residuals - linear model with true error parameter")
# plt.show()

plt.figure()
plt.scatter(eth["search_interest"], eth["Close"])
plt.plot(xs, get_linear_model(minimization_linear.x, xs), label='Without error parameter')
plt.plot(xs, get_linear_model(minimization_linear_sys.x[:-1], xs), label='With error parameter')
plt.plot(xs, get_linear_model(minimization_linear_true.x[:-1], xs), label='With true error parameter')
plt.xlabel("Search interest")
plt.ylabel("Closing price (USD)")
plt.title("Linear models (optimal estimation)")
plt.legend()
plt.savefig("Linear models (optimal estimation).png")
plt.show()

mcmc_linear = MCMC.MCMC(get_linear_model, 2)
mcmc_linear.MCMC(minimization_linear_sys.x[:-1], [0.1, 0.1], interest, closing, err, 10000)
diagnostics.print_diagnostics(mcmc_linear, "Linear MCMC - without error parameter", False)
mcmc_linear.calculate_confidence_intervals(4000)

# the last element of the median params array is the corresponding chi-squared
print("MCMC Linear\n\tConfidence intervals: {0}\n\tMedian parameters: {1}".format(mcmc_linear.confidence_intervals, mcmc_linear.median_parameters))
diagnostics.print_random_models(mcmc_linear.get_samples(), get_linear_model, interest, closing, "linear, no error parameter", False)


# if you don't give the optimal estimate as the starting set of parameter values for MCMC - fake convergence
mcmc_error_linear = MCMC_error.MCMC(get_linear_model, 3)
mcmc_error_linear.MCMC(np.array([100.0, 100.0, 100.0]), [0.5, 0.5, 0.5], interest, closing, 30000)
diagnostics.print_diagnostics(mcmc_error_linear, "Linear MCMC - with error parameter - non-optimal starting point", True)
mcmc_error_linear.calculate_confidence_intervals(10000)
print("MCMC Error Linear\n\tConfidence intervals: {0}\n\tMedian parameters: {1}".format(mcmc_error_linear.confidence_intervals, mcmc_error_linear.median_parameters))
diagnostics.print_random_models(mcmc_error_linear.get_samples(), get_linear_model, interest, closing, "linear, error parameter", True)

# and if you do - no convergence
mcmc_error_linear_opt = MCMC_error.MCMC(get_linear_model, 3)
mcmc_error_linear_opt.MCMC(minimization_linear_sys.x, [0.5, 0.5, 0.5], interest, closing, 100000)
diagnostics.print_diagnostics(mcmc_error_linear_opt, "Linear MCMC - with error parameter - optimal starting point", True)
mcmc_error_linear_opt.calculate_confidence_intervals(80000)
print("MCMC Error Linear Opt\n\tConfidence intervals: {0}\n\tMedian parameters: {1}".format(mcmc_error_linear_opt.confidence_intervals, mcmc_error_linear_opt.median_parameters))
diagnostics.print_random_models(mcmc_error_linear_opt.get_samples(), get_linear_model, interest, closing, "linear, error parameter - combined err", True)

print("Diagnostics - Linear models")
print("Optimal estimation results")
print("No error parameter\n\ty = {} + {} * x".format(minimization_linear.x[0], minimization_linear.x[1]))
print("Error parameter\n\ty = {0} + {1} * x\n\tError: {2}".format(minimization_linear_sys.x[0], minimization_linear_sys.x[1], minimization_linear_sys.x[2]))
print("True error parameter\n\ty = {0} + {1} * x\n\tError: {2}".format(minimization_linear_true.x[0], minimization_linear_true.x[1], minimization_linear_true.x[2]))
print("MCMC results")
print("No error parameter\n\ty = {} + {} * x".format(mcmc_linear.chain[-1][0], mcmc_linear.chain[-1][1]))
print("Error parameter\n\ty = {0} + {1} * x\n\tError: {2}".format(mcmc_error_linear.chain[-1][0], mcmc_error_linear.chain[-1][1], mcmc_error_linear.chain[-1][2]))

# model fitting (for reference, pretty much unused)
# how to read poly1d objs: poly1d([ a, b, c]) = ax^2 + bx + c
model_linear = np.poly1d(np.polyfit(interest, closing, 1))
model_quad = np.poly1d(np.polyfit(interest, closing, 2))
model_cubic = np.poly1d(np.polyfit(interest, closing, 3))

# now let's try the same thing but with the quadratic stuff-- this doesn't estimate error------------------
parameter_guess_quad = model_quad.coef[::-1]
minimization_quad = op.minimize(negative_ln_likelihood_quad, parameter_guess_quad, args=(interest, closing, err))

# estimate error
parameter_guess_quad_sys = np.array([1, 1, 1, 1])
minimization_quad_sys = op.minimize(negative_ln_likelihood_sys_quad, parameter_guess_quad_sys, args=(interest, closing))

mcmc_quad = MCMC.MCMC(get_quadratic_model, 3)
mcmc_quad.MCMC(model_quad.coef[::-1], [0.15, 0.15, 0.15], interest, closing, err, 50000)
diagnostics.print_diagnostics(mcmc_quad, "Quadratic MCMC - without error parameter", False)
mcmc_quad.calculate_confidence_intervals(20000)
print("MCMC Quad\n\tConfidence intervals: {0}\n\tMedian parameters: {1}".format(mcmc_quad.confidence_intervals, mcmc_quad.median_parameters))
diagnostics.print_random_models(mcmc_quad.get_samples(), get_quadratic_model, interest, closing, "quadratic, no error parameter", False)

mcmc_error_quad = MCMC_error.MCMC(get_quadratic_model, 4)
mcmc_error_quad.MCMC(minimization_quad_sys.x, [0.5, 0.5, 0.5, 0.5], interest, closing, 50000)

# virtually incapable of convergence
mcmc_error_quad_take_two = MCMC_error.MCMC(get_quadratic_model, 4)
mcmc_error_quad_take_two.MCMC(minimization_quad_sys.x, [1.5, 0.7, 0.7, 1], interest, closing, 50000)
diagnostics.print_diagnostics(mcmc_error_quad_take_two, "Quadratic MCMC - with error parameter - optimal starting point", True)
mcmc_error_quad_take_two.calculate_confidence_intervals(20000)
print("MCMC Error Quad\n\tConfidence intervals: {0}\n\tMedian parameters: {1}".format(mcmc_error_quad_take_two.confidence_intervals, mcmc_error_quad_take_two.median_parameters))
diagnostics.print_random_models(mcmc_error_quad_take_two.get_samples(), get_quadratic_model, interest, closing, "quadratic, error parameter", True)

plt.figure()
plt.scatter(eth["search_interest"], eth["Close"], label='Data')
plt.plot(xs, get_quadratic_model(minimization_quad.x, xs), label='Without error term', color='red')
plt.plot(xs, get_quadratic_model(minimization_quad_sys.x[:-1], xs), label='With error term', color='green')
plt.xlabel("Search interest")
plt.ylabel("Closing price (USD)")
plt.title("Quadratic models (optimal estimation)")
plt.legend()
plt.savefig("Quadratic models (optimal estimation).png")
plt.show()

# onto cubic shit, I guess----------------------------------------------
parameter_guess_cubic = model_cubic.coef[::-1]
minimization_cubic = op.minimize(negative_ln_likelihood_cubic, parameter_guess_cubic, args=(interest, closing, err))

parameter_guess_cubic_sys = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
minimization_cubic_sys = op.minimize(negative_ln_likelihood_sys_cubic, parameter_guess_cubic_sys, args=(interest, closing))

plt.figure()
plt.scatter(eth["search_interest"], eth["Close"], label='Data')
# plt.plot(xs, model_cubic(xs), label='Cubic model', color='red')
# plt.plot(xs, get_cubic_model(mcmc_cubic.chain[-1], xs), label='MCMC cubic model', color='green')
plt.plot(xs, get_cubic_model(minimization_cubic.x, xs), label='Without error term', color='red')
# plt.plot(xs, get_cubic_model(minimization_cubic_sys.x[:-1], xs), label='With error term', color='green')
plt.plot(xs, get_cubic_model(model_cubic.coef[::-1], xs), label='With error term', color='green')
plt.xlabel("Search interest")
plt.ylabel("Closing price (USD)")
plt.title("Cubic models (optimal estimation)")
plt.legend()
plt.savefig("Cubic models (optimal estimation).png")
plt.show()

mcmc_cubic = MCMC.MCMC(get_cubic_model, 4)
mcmc_cubic.MCMC(model_cubic.coef[::-1], [0.5, 0.2, 0.02, 0.02], interest, closing, err, 20000)
diagnostics.print_diagnostics(mcmc_cubic, "Cubic MCMC - without error parameter", False)
mcmc_cubic.calculate_confidence_intervals(17500)
print("MCMC Cubic\n\tConfidence intervals: {0}\n\tMedian parameters: {1}".format(mcmc_cubic.confidence_intervals, mcmc_cubic.median_parameters))
diagnostics.print_random_models(mcmc_cubic.get_samples(), get_cubic_model, interest, closing, "cubic, no error parameter", False)

mcmc_error_cubic = MCMC_error.MCMC(get_cubic_model, 5)
mcmc_error_cubic_params0 = np.append(model_cubic.coef[::-1], 100)
mcmc_error_cubic.MCMC(mcmc_error_cubic_params0, [1.5, 1.5, 0.015, 0.0015, 0.2], interest, closing, 120000)
diagnostics.print_diagnostics(mcmc_error_cubic, "Cubic MCMC - with error parameter - optimal starting point", True)
mcmc_error_cubic.calculate_confidence_intervals(60000)
print("MCMC Error Cubic\n\tConfidence intervals: {0}\n\tMedian parameters: {1}".format(mcmc_error_cubic.confidence_intervals, mcmc_error_cubic.median_parameters))
diagnostics.print_random_models(mcmc_error_cubic.get_samples(), get_cubic_model, interest, closing, "cubic, error parameter", True)

# sigmoid fit-----------------------------------------------------------
def sigmoid(x, L, x0, k, b):
    return L / (1 + np.exp(-k * (x - x0))) + b

# initial guess
params0 = [np.max(closing), np.median(interest), 1, np.min(closing)]
popt, pcov = op.curve_fit(sigmoid, interest, closing, params0, method='dogbox')

# minimizer
minimization_sigmoid = op.minimize(negative_ln_likelihood_sigmoid, params0, args=(interest, closing, err))
minimization_sigmoid_sys = op.minimize(negative_ln_likelihood_sys_sigmoid, parameter_guess_cubic_sys, args=(interest, closing))

# sigmoid(xs, *popt) is equivalent to get_sigmoid_model(popt, xs)
plt.figure()
plt.scatter(eth["search_interest"], eth["Close"], label='Data')
# plt.plot(xs, get_sigmoid_model(minimization_sigmoid.x, xs), label='Without error term', color='red')
plt.plot(xs, get_sigmoid_model(popt, xs), label='With error term', color='black')
# plt.plot(xs, get_sigmoid_model(mcmc_sigmoid.chain[-1], xs), label='MCMC sigmoid model', color='green')
# plt.plot(xs, sigmoid(xs, *minimization_sigmoid.x), label='Minimizer', color='green')
# plt.plot(xs, sigmoid(xs, *minimization_sigmoid_sys.x[:-1]), label='Minimizer with error', color='black')
plt.xlabel("Search interest")
plt.ylabel("Closing price (USD)")
plt.title("Sigmoid model (optimal estimation)")
plt.legend()
plt.savefig("Sigmoid model (optimal estimation).png")
plt.show()

mcmc_sigmoid = MCMC.MCMC(get_sigmoid_model, 4)
mcmc_sigmoid.MCMC(popt, [15, 0.15, 0.015, 1.5], interest, closing, err, 15000)

# oh these look awful
diagnostics.print_diagnostics(mcmc_sigmoid, "Sigmoid MCMC - without error parameter - optimal starting point", False)
mcmc_sigmoid.calculate_confidence_intervals(6000)
print("MCMC Sigmoid\n\tConfidence intervals: {0}\n\tMedian parameters: {1}".format(mcmc_sigmoid.confidence_intervals, mcmc_sigmoid.median_parameters))
diagnostics.print_random_models(mcmc_sigmoid.get_samples(), get_sigmoid_model, interest, closing, "sigmoid, no error parameter", False)

mcmc_sigmoid_error = MCMC_error.MCMC(get_sigmoid_model, 5)
sigmoid_params0 = np.append(popt, 100)
mcmc_sigmoid_error.MCMC(sigmoid_params0, [15, 0.15, 0.015, 1.5, 1], interest, closing, 10000)
diagnostics.print_diagnostics(mcmc_sigmoid_error, "Sigmoid MCMC - with error parameter - optimal starting point", True)
mcmc_sigmoid_error.calculate_confidence_intervals(6000)
print("MCMC Error Sigmoid\n\tConfidence intervals: {0}\n\tMedian parameters: {1}".format(mcmc_sigmoid_error.confidence_intervals, mcmc_sigmoid_error.median_parameters))
diagnostics.print_random_models(mcmc_sigmoid_error.get_samples(), get_sigmoid_model, interest, closing, "sigmoid, error parameter", True)