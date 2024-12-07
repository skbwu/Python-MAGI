import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

##############################
# Define the ODE system
##############################
def f_vec(t, X, thetas):
    '''
    Log-scale EIR model with E implicitly represented.
    ODE:
      dE/dt = beta * S * I - sigma * E
      dI/dt = sigma * E - gamma * I
      dR/dt = gamma * I
    where S = 1 - E - I - R

    On the log scale: X = [logE, logI, logR]

    Parameters
    ----------
    t : float
        Time
    X : np.array (3,)
        [logE, logI, logR]
    thetas : array-like
        [beta, gamma, sigma]

    Returns
    -------
    dXdt : np.array (3,)
        Derivatives in log scale
    '''
    logE, logI, logR = X
    beta, gamma, sigma = thetas

    E = np.exp(logE)
    I = np.exp(logI)
    R = np.exp(logR)

    S = 1.0 - E - I - R
    if S <= 0.0:
        S = 1e-10  # small positive value to avoid instability

    # Original scale derivatives
    dEdt = beta * S * I - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I

    # Log scale derivatives
    dlogEdt = dEdt / E
    dlogIdt = dIdt / I
    dlogRdt = dRdt / R

    return np.array([dlogEdt, dlogIdt, dlogRdt])

def ODE_log_scale(t, y, theta):
    return f_vec(t, y, theta)

##############################
# Negative log-likelihood function
##############################
# Negative log-likelihood function
def negative_log_likelihood(params, ts_obs, logI_obs, logR_obs):
    # params = [log_beta, log_gamma, log_sigma, logE0, logI0, logR0, log_sigma_obs]
    log_beta, log_gamma, log_sigma, logE0, logI0, logR0, log_sigma_obs = params

    beta = np.exp(log_beta)
    gamma = np.exp(log_gamma)
    sigma = np.exp(log_sigma)
    sigma_obs = np.exp(log_sigma_obs)

    thetas = [beta, gamma, sigma]
    X0 = np.array([logE0, logI0, logR0])

    # Solve ODE
    sol = solve_ivp(lambda t, y: ODE_log_scale(t, y, thetas),
                    t_span=(ts_obs[0], ts_obs[-1]),
                    y0=X0, t_eval=ts_obs, rtol=1e-10, atol=1e-10)

    if not sol.success:
        return np.inf

    X_pred = sol.y.T
    logI_pred = X_pred[:, 1]
    logR_pred = X_pred[:, 2]

    mask_I = ~np.isnan(logI_obs)
    mask_R = ~np.isnan(logR_obs)

    residuals_I = (logI_obs[mask_I] - logI_pred[mask_I])**2
    residuals_R = (logR_obs[mask_R] - logR_pred[mask_R])**2

    SSE = np.sum(residuals_I) + np.sum(residuals_R)
    N = np.sum(mask_I) + np.sum(mask_R)

    # Negative log-likelihood with unknown sigma_obs:
    # NLL = N*log(sigma_obs) + SSE/(2*sigma_obs^2) + constant
    # The constant doesn't affect optimization.
    NLL = N * log_sigma_obs + SSE / (2.0 * sigma_obs**2)

    return NLL

# Main function for running optimization and visualization
def mle(ts_obs, X_obs_full, maxiter=1000):
    logI_obs = X_obs_full[:, 1]
    logR_obs = X_obs_full[:, 2]

    # Initial guesses for parameters
    init_log_beta = 0.5 * np.random.randn()
    init_log_gamma = 0.5 * np.random.randn()
    init_log_sigma = 0.5 * np.random.randn()
    init_logI0 = logI_obs[0] if not np.isnan(logI_obs[0]) else -4.0
    init_logR0 = logR_obs[0] if not np.isnan(logR_obs[0]) else -4.0
    init_logE0 = -4.0
    init_log_sigma_obs = np.log(0.1)  # starting guess for sigma_obs

    init_guess = [init_log_beta, init_log_gamma, init_log_sigma,
                  init_logE0, init_logI0, init_logR0, init_log_sigma_obs]

    # Adjust bounds if needed. For sigma_obs, let's allow it to vary widely:
    bounds = [(-5, 5), (-3, 3), (-3, 3), (-10, 0), (-10, 5), (-10, 5), (-5, 0)]
    # Here, (-5,0) for log_sigma_obs means sigma_obs is between exp(-5)~0.0067 and exp(0)=1.0. Adjust as needed.

    res = minimize(negative_log_likelihood, init_guess,
                   args=(ts_obs, logI_obs, logR_obs),
                   method='L-BFGS-B', bounds=bounds, options={'maxiter': maxiter})

    log_beta_est, log_gamma_est, log_sigma_est, logE0_est, logI0_est, logR0_est, log_sigma_obs_est = res.x

    beta_est = np.exp(log_beta_est)
    gamma_est = np.exp(log_gamma_est)
    sigma_est = np.exp(log_sigma_est)
    sigma_obs_est = np.exp(log_sigma_obs_est)

    print("Best fit parameters (on natural scale):")
    print("beta:         ", beta_est)
    print("gamma:        ", gamma_est)
    print("sigma:        ", sigma_est)
    print("E0:           ", np.exp(logE0_est))
    print("I0:           ", np.exp(logI0_est))
    print("R0:           ", np.exp(logR0_est))
    print("sigma_obs:    ", sigma_obs_est)

    final_thetas = [beta_est, gamma_est, sigma_est]
    X0_final = np.array([logE0_est, logI0_est, logR0_est])

    # Compute approximate uncertainty
    hess_inv_mat = res.hess_inv.todense()  # inverse Hessian approximation
    se = np.sqrt(np.diag(hess_inv_mat))  # standard errors on log-scale
    z_val = 1.96
    lower_log = res.x - z_val * se
    upper_log = res.x + z_val * se

    param_names = ["beta", "gamma", "sigma", "E0", "I0", "R0", "sigma_obs"]
    print("\nApproximate 95% CI (on log scale):")
    for i, p in enumerate(param_names):
        print(f"{p}: [{lower_log[i]:.4f}, {upper_log[i]:.4f}]")

    lower_nat = np.exp(lower_log)
    upper_nat = np.exp(upper_log)
    print("\nApproximate 95% CI (on natural scale):")
    for i, p in enumerate(param_names):
        print(f"{p}: [{lower_nat[i]:.4f}, {upper_nat[i]:.4f}]")

    return final_thetas, X0_final, sigma_obs_est, res.fun, (res.x, se, lower_nat, upper_nat), res

##############################
# Metropolis-Hastings Sampler
##############################
def metropolis_hastings(initial_params, n_samples, proposal_cov, ts_obs, logI_obs, logR_obs):
    # log_posterior is proportional to -NLL, so we can just use negative_log_likelihood and convert
    # For MH, we actually only need NLL and use exp(-NLL)
    # We'll store samples in a chain:
    chain = np.zeros((n_samples, len(initial_params)))
    chain[0] = initial_params

    current_params = initial_params.copy()
    current_nll = negative_log_likelihood(current_params, ts_obs, logI_obs, logR_obs)
    accepted = 0

    for i in range(1, n_samples):
        # Propose new parameters
        proposal = np.random.multivariate_normal(current_params, proposal_cov)
        # Compute NLL for proposal
        proposal_nll = negative_log_likelihood(proposal, ts_obs, logI_obs, logR_obs)

        # Metropolis criterion
        # alpha = exp(-(proposal_nll - current_nll)) = exp(-proposal_nll)/exp(-current_nll)
        # If alpha >= 1, accept. If alpha < 1, accept with probability alpha.
        alpha = np.exp(-(proposal_nll - current_nll))

        if np.random.rand() < alpha:
            # Accept
            current_params = proposal
            current_nll = proposal_nll
            accepted += 1

        chain[i] = current_params

    acceptance_rate = accepted / (n_samples - 1)
    return chain, acceptance_rate
