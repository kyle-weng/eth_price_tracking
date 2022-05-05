import numpy as np
from typing import Tuple

class MCMC:
    def __init__(self, get_model, dim: int) -> None:
        self.dim = dim
        self.get_model = get_model
        self.chain, self.accept_chain, self.chi_squared_chain = None, None, None
        self.bayesian_information_criterion = None
        self.confidence_intervals = []

    # obtain the chi squared value of the model y-values given current parameters vs. the measured y-values
    def get_log_likelihood(self, params: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
        error = params[-1]
        chi_squared = np.sum(np.power((self.get_model(params, x) - y) / error, 2))
        return chi_squared

    # select a model parameter to perturb-- picks the literal number (as opposed to the name of the parameter)
    def perturb_pick(self, params: np.ndarray) -> Tuple[int, float]:
        # pick a parameter
        assert len(params.shape) == 1 # guarantees params has shape (n,), which is equivalent to (1, n)
        param_index = np.random.randint(0, params.shape[0])
        param_to_perturb = params[param_index]
        return param_index, param_to_perturb
        # return 2, 0 # force only error to be changed

    # obtain a trial model parameter for the current step
    # clarification: perturbed_value hasn't actually been perturbed yet. jfc the variable naming here is atrocious
    def propose_param(self, active_param: np.ndarray, step_size: np.ndarray, perturbed_value_index: int) -> np.ndarray:
        current_param = active_param.copy()
        value_to_perturb = active_param[perturbed_value_index]
        
        perturbed_value = value_to_perturb + np.random.normal(0, step_size[perturbed_value_index])
        current_param[perturbed_value_index] = perturbed_value

        # assert not np.array_equal(current_param, active_param)
        return current_param
        
    #evaluate whether to step to the new trial value
    def step_eval(self, params: np.ndarray, step_size: np.ndarray, x: np.ndarray, \
                y: np.ndarray, perturbed_value_index: int) -> Tuple[np.ndarray, np.ndarray]:
        accepted = np.array([None] * self.dim)
        # param_index = np.where(params == perturbed_value)[0][0]
        value_to_perturb = params[perturbed_value_index]

        # the chi squared value of the parameters from the previous step
        chi_squared_old = self.get_log_likelihood(params, x, y)
        
        # read in the trial model parameters for the current step-- ie. try the perturbation
        try_param = self.propose_param(params, step_size, perturbed_value_index)

        # assert not np.array_equal(try_param, params)

        # the chi squared value of the trial model parameters for the current step
        chi_squared_new = self.get_log_likelihood(try_param, x, y)

        # calculate ratio of posterior probabilities
        # ratio = p_new/p_old // np.longdouble cast is to avoid overflow (not that it matters lol)
        ratio = np.exp(-0.5 * (chi_squared_new - chi_squared_old))

        # default "return values"
        accepted[perturbed_value_index] = False
        new_param = params
        chi_squared = chi_squared_old

        # print(ratio, chi_squared_new, chi_squared_old)
        if ratio > 1:
            # implied p_new > p_old-- we're stepping in the right direction-- accept step
            accepted[perturbed_value_index] = True
            new_param = try_param
            chi_squared = chi_squared_new
        else:
            # lmao this ain't it-- probability time!
            draw = np.random.uniform(0, 1)
            if draw > ratio:
                # reject
                pass
            else:
                # accept
                accepted[perturbed_value_index] = True
                new_param = try_param
                chi_squared = chi_squared_new
            
        return new_param, accepted, chi_squared

    def MCMC(self, params: np.ndarray, step_size: np.ndarray, x: np.ndarray, \
            y: np.ndarray, n_steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # I guess we're just making the chains here lol

        # records parameter values
        chain = np.zeros((n_steps, self.dim))

        # records booleans (equivalently, 0/1 ints) corresponding to whether a proposal was accepted
        accept_chain = np.zeros((n_steps, self.dim))

        # records chi squared values
        chi_squared_chain = np.zeros(n_steps)

        for i in range(0, n_steps):
            param_index, _ = self.perturb_pick(params)
            new_params, accepted, chi_squared = self.step_eval(params, step_size, x, y, param_index)
            accepted_final = True in accepted

            # update chain
            if accepted_final:
                assert not np.array_equal(new_params, params)
                chain[i] = new_params
                params = new_params
            else:
                chain[i] = params
            
            # update accept chain
            accept_chain[i] = accepted

            # update chi squared chain
            chi_squared_chain[i] = chi_squared

        self.chain = chain
        self.accept_chain = accept_chain
        self.chi_squared_chain = chi_squared_chain

        # calculate BIC on the side
        self.bayesian_information_criterion = self.dim * np.log(len(x)) - 2 * chi_squared_chain[-1]

        # calculate 68% CIs for all parameters on the side
        stacked_chain = np.column_stack((chain, chi_squared_chain))
        stacked_chain_truncated = stacked_chain[: int(len(stacked_chain) * 0.68)]
        for i in range(0, self.dim):
            eligible_parameter_values = stacked_chain_truncated[:, i]
            self.confidence_intervals.append((np.min(eligible_parameter_values), np.max(eligible_parameter_values)))

        return chain, accept_chain, chi_squared_chain