""" gp.py
Bayesian optimisation of loss functions.
"""

import numpy as np
import sklearn.gaussian_process as gp

from scipy.stats import norm
from scipy.optimize import minimize
from work1 import *

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

def expected_improvement(x, gaussian_process, evaluated_loss, greater_is_better=False, n_params=1):
	""" expected_improvement
	Expected improvement acquisition function.
	Arguments:
	----------
	x: array-like, shape = [n_samples, n_hyperparams]
		The point for which the expected improvement needs to be computed.

	gaussian_process: GaussianProcessRegressor object.
		Gaussian process trained on previously evaluated hyperparameters.

	evaluated_loss: Numpy array.
		Numpy array that contains the values off the loss function for the previously
		evaluated hyperparameters.
        
	greater_is_better: Boolean.
		Boolean flag that indicates whether the loss function is to be maximised or minimised.

	n_params: int.
		Dimension of the hyperparameter space.
	"""

	x_to_predict = x.reshape(-1, n_params)

	mu, sigma = gaussian_process.predict(x_to_predict, return_std=True)

	if greater_is_better:
		loss_optimum = np.max(evaluated_loss)
	else:
		loss_optimum = np.min(evaluated_loss)

	scaling_factor = (-1) ** (not greater_is_better)

	# In case sigma equals zero
	with np.errstate(divide='ignore'):
		Z = scaling_factor * (mu - loss_optimum) / sigma
		expected_improvement = scaling_factor * (mu - loss_optimum) * norm.cdf(Z) + sigma * norm.pdf(Z)
		expected_improvement[sigma == 0.0] == 0.0

	return -1 * expected_improvement

def sample_next_hyperparameter(acquisition_func, gaussian_process, evaluated_loss, greater_is_better=False,
                               bounds=(0, 10), n_restarts=25):
	""" 
	sample_next_hyperparameter
	Proposes the next hyperparameter to sample the loss function for.	
	Arguments:
	----------
	acquisition_func: function.
	Acquisition function to optimise.

	gaussian_process: GaussianProcessRegressor object.
		Gaussian process trained on previously evaluated hyperparameters.

	evaluated_loss: array-like, shape = [n_obs,]
		Numpy array that contains the values off the loss function for the previously
		evaluated hyperparameters.

	greater_is_better: Boolean.
		Boolean flag that indicates whether the loss function is to be maximised or minimised.

	bounds: Tuple.
		Bounds for the L-BFGS optimiser.
	n_restarts: integer.
	Number of times to run the minimiser with different starting points.
	"""
	best_x = None
	best_acquisition_value = 1
	n_params = bounds.shape[0]

	for starting_point in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, n_params)):

		res = minimize(fun=acquisition_func,
			x0=starting_point.reshape(1, -1),
			bounds=bounds,
			method='L-BFGS-B',
			args=(gaussian_process, evaluated_loss, greater_is_better, n_params))

		if res.fun < best_acquisition_value:
			best_acquisition_value = res.fun
			best_x = res.x

		return best_x


def bayesian_optimisation(n_iters, v, sample_loss, bounds, x0=None, n_pre_samples=5,
                          gp_params=None, random_search=False, alpha=1e-5, epsilon=1e-7):
	""" 
	bayesian_optimisation
	Uses Gaussian Processes to optimise the loss function `sample_loss`.
	Arguments:
	----------
	n_iters: integer.
		Number of iterations to run the search algorithm.

	sample_loss: function.
		Function to be optimised.

	bounds: array-like, shape = [n_params, 2].
		Lower and upper bounds on the parameters of the function `sample_loss`.

	x0: array-like, shape = [n_pre_samples, n_params].
		Array of initial points to sample the loss function for. If None, randomly
		samples from the loss function.

	n_pre_samples: integer.
		If x0 is None, samples `n_pre_samples` initial points from the loss function.
	gp_params: dictionary.
		Dictionary of parameters to pass on to the underlying Gaussian Process.

	random_search: integer.
		Flag that indicates whether to perform random search or L-BFGS-B optimisation
		over the acquisition function.

	alpha: double.
		Variance of the error term of the GP.

	epsilon: double.
		Precision tolerance for floats.
	"""

	x_list = []
	y_list = []

	n_params = bounds.shape[0]

	if x0 is None:
		for params in np.random.uniform(bounds[:, 0], bounds[:, 1], (n_pre_samples, bounds.shape[0])):
			x_list.append(params)
			y_list.append(sample_loss(params))
	else:
		for params in x0:
			x_list.append(params)
		for p in sample_loss:
			y_list.append(p)

	xp = np.array(x_list)
	yp = np.array(y_list)

	# Create the GP
	if gp_params is not None:
		model = gp.GaussianProcessRegressor(**gp_params)
	else:
		kernel = gp.kernels.Matern()
		model = gp.GaussianProcessRegressor(kernel=kernel,
							alpha=alpha,
							n_restarts_optimizer=10,
							normalize_y=True)

	predictions = open('gaussian_prediction.txt', 'w')
	for n in range(n_iters):

		model.fit(xp, yp)

		# Sample next hyperparameter
		if random_search:
			x_random = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(random_search, n_params))
			ei = -1 * expected_improvement(x_random, model, yp, greater_is_better=True, n_params=n_params)
			next_sample = x_random[np.argmax(ei), :]
		else:
			next_sample = sample_next_hyperparameter(expected_improvement, model, yp, greater_is_better=False, bounds=bounds, n_restarts=100)
		
		print(next_sample)
		
		# Duplicates will break the GP. In case of a duplicate, we will randomly sample a next query point.
		if np.any(np.abs(next_sample - xp) <= epsilon):
			next_sample = np.random.uniform(bounds[:, 0], bounds[:, 1], bounds.shape[0])

		# Sample loss for new set of parameters
		#cv_score = sample_loss

		cv_score = v.init_setup(n, next_sample)
		fileName = 'iter_%d' %(n)
		extract = open('%s/OSZICAR' %(fileName), 'r')
		lines = extract.readlines()
		line = lines[-2]
		line = line.strip().split()
		cv_score = float(line[2])
		
		# Extract value from `tail -n 2 OSZICAR | awk '{print $3}' `
		# Update lists
		x_list.append(next_sample)
		y_list.append(cv_score)

		# Update xp and yp
		xp = np.array(x_list)
		yp = np.array(y_list)

		x = np.array(x_list[-1]).reshape(-1,4)

		mu,sigma = model.predict(x, return_std=True)
		writePred = open('%s/prediction.txt' %(fileName), 'w')
		writePred.write('mu = %12.6f, ground_truth = %12.6f,  std = %12.6f \n' %(mu, y_list[-1], sigma))
		writePred.close()	
		#predictions.write('%12.6f %12.6f %12.6f %12.6f      %12.6f   %12.6f  %12.6f\n' %(x[0][0], x[0][1], x[0][2], x[0][3], y_list[-1], mu, sigma))

	return xp, yp, model

x0 = [[0.637536,0.791126,0.836563,0.654109],
[0.529659,0.616956,0.546202,0.658692],
[0.740083,0.587217,0.899518,0.569349],
[0.274015,0.784627,0.405341,0.644914],
[0.218928,0.246003,0.963137,0.477641],
[0.431478,0.940593,0.947795,0.721674],
[0.834946,0.663310,0.340715,0.133536],
[0.065144,0.701458,0.470773,0.846166],
[0.805929,0.945046,0.054464,0.772177],
[0.774392,0.403658,0.070715,0.591531],
[0.806566,0.209778,0.383663,0.397396],
[0.296359,0.475026,0.936742,0.858693],
[0.998724,0.235684,0.065597,0.933679],
[0.830873,0.616156,0.239211,0.318329],
[0.102256,0.882374,0.517792,0.403878],
[0.305194,0.920399,0.907332,0.177889],
[0.847481,0.835709,0.827204,0.468648],
[0.720034,0.513515,0.784034,0.109078],
[0.460173,0.027772,0.857058,0.607143]]

#sample_loss = [-1482.0191,
sample_loss = [-1482.0191,-1482.0203,-1485.0651,-1485.0672,-1482.8304,
-1485.3641,-1482.8357,-1485.0687,-1482.0219,-1485.3663,-1485.0672,-1482.8342,
-1485.0652,-1485.3639,-1482.8388,-1485.0650,-1485.0690,-1485.3639,-1485.0664]

bounds = [[0.0,1.0],[0.0,1.0],[0.0,1.0],[0.0,1.0]]
bounds = np.array(bounds)

v= vasp()
v.run()
v.Dfs()

(xp,yp,gaussian_process) = bayesian_optimisation(30, v, sample_loss, bounds, x0, n_pre_samples=19, random_search=False)
print(xp)
print('\n\n\n')
print(yp)
print('\n\n\n')
print(gaussian_process)

x_to_predict = xp.reshape(-1,4)

#x_to_predict = np.array([0.637536,0.791126,0.836563,0.654109]).reshape(-1,4)
mu, sigma = gaussian_process.predict(x_to_predict, return_std=True)

print('-------------mu------------')
print(mu)
print('------------sigma----------')
print(sigma)
print('---------- mu - sigma -----')
print(mu - sigma)
print('---------- mu + sigma -----')
print(mu + sigma)

#plt.plot(x, mu, 'o',label="GP mean")
#plt.fill_between(x, mu-sigma, mu+sigma, alpha=0.9)
#plt.show()


