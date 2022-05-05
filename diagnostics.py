from matplotlib import pyplot as plt
import MCMC, MCMC_error
import numpy as np

def print_diagnostics(model: MCMC.MCMC, name: str, has_error: bool):
    plt.figure()
    plt.plot(model.chi_squared_chain)
    plt.title(name)
    plt.xlabel("Iteration")
    plt.ylabel("$\chi^2$")
    plt.savefig("{0}.png".format(name))
    plt.show()

    char_ascii = 96

    for i in range(0, model.dim):
        plt.figure()
        plt.plot(model.chain[:, i])
        plt.title(name)
        plt.xlabel("Iteration")
        if has_error and i == model.dim - 1:
            plt.ylabel("error")
            plt.savefig("{0} - {1}.png".format(name, "error"))
        else:
            char_ascii += 1
            plt.ylabel(chr(char_ascii))
            plt.savefig("{0} - {1}.png".format(name, chr(char_ascii)))
        plt.show()
    
    print("Acceptance ratio:", model.acceptance_ratio)
    print("BIC:", model.bayesian_information_criterion)
    print("Remember to pick a burn-in point for trimming and recalculate confidence intervals")

def print_random_models(parameters: np.ndarray, get_model, x: np.ndarray, y: np.ndarray, model_type: str, has_error: bool):
    xs = np.linspace(np.min(x), np.max(x), 500)

    plt.figure()
    for i in range(0, len(parameters)):
        if has_error:
            plt.plot(xs, get_model(parameters[i][:-1], xs), color='black')
        else:
            plt.plot(xs, get_model(parameters[i], xs), color='black')
    plt.scatter(x, y)
    plt.xlabel("Search interest")
    plt.ylabel("Closing price (USD)")
    plt.title("Model predictions ({})".format(model_type))
    plt.savefig("Model predictions ({}).png".format(model_type))
    plt.show()