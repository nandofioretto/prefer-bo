import numpy as np
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(0, '/home/fioretto/Repos/preference_elicitation/')

from bayes_opt import BayesianOptimization

# use sklearn's default parameters for theta and random_start
gp_params = {"alpha": 1e-5, "n_restarts_optimizer": 2}

np.random.seed(42)
xs = np.linspace(-2, 10, 10000)
f = np.exp(-(xs - 2)**2) + np.exp(-(xs - 6)**2/10) + 1/ (xs**2 + 1)

def plot_bo(f, bo):
    xs = [x["x"] for x in bo.res["all"]["params"]]
    ys = bo.res["all"]["values"]

    mean, sigma = bo.gp.predict(np.arange(len(f)).reshape(-1, 1), return_std=True)

    plt.figure(figsize=(16, 9))
    plt.plot(f)
    plt.plot(np.arange(len(f)), mean)
    plt.fill_between(np.arange(len(f)), mean + sigma, mean - sigma, alpha=0.1)
    plt.scatter(bo.X.flatten(), bo.Y, c="red", s=50, zorder=10)
    plt.xlim(0, len(f))
    plt.ylim(f.min() - 0.1 * (f.max() - f.min()), f.max() + 0.1 * (f.max() - f.min()))
    plt.show()



# bo = BayesianOptimization(f=lambda x: f[int(x)],
#                           pbounds={"x": (0, len(f)-1)},
#                           verbose=1)
# bo.maximize(init_points=2, n_iter=25, acq="ucb", kappa=1, **gp_params)
# plot_bo(f, bo)


bo = BayesianOptimization(f=lambda x: f[int(x)],
                          pbounds={"x": (0, len(f)-1)},
                          verbose=1)

n_eval = 25
bo.maximize(init_points=2, n_iter=n_eval, acq="ucb", kappa=10, **gp_params)
#            noise_sd=[0.001 * i for i in range(n_eval)])

plot_bo(f, bo)