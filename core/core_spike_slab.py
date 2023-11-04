import os
import numpy as np
import scipy.stats as stats
import random
import tensorflow as tf
# from numpy import tf.linalg.matmul, tf.eye, log
# from numpy.linalg import tf.linalg.inv, tf.linalg.det, tf.linalg.lstsq
from scipy.linalg import solve as linsolve
from scipy.linalg import norm
import scipy.special as sps
from helper import *
from truncated_multivariate_normal import *

class Data:
	def __init__(self, D, y):
		self.D = D
		self.y = y

class Params:
	def __init__(self, N, P, a_v, b_v, a_p, b_p, a_sigma, b_sigma):
		self.N = N
		self.P = P
		self.a_v = a_v
		self.b_v = b_v
		self.a_p = a_p
		self.b_p = b_p
		self.a_sigma = a_sigma
		self.b_sigma = b_sigma

class State:
	def __init__(self, params):
		self.p0 = 0.
		self.vs = 0.
		self.sig2 = 0.
		self.z = tf.zeros(params.P, dtype=int)
		self.theta = tf.zeros(params.P)

class Chain:
	def __init__(self, params, chain_length, burn):
		self.p0 = tf.zeros(chain_length)
		self.vs = tf.zeros(chain_length)
		self.sig2 = tf.zeros(chain_length)
		self.z = tf.zeros([chain_length,params.P], dtype=int)
		self.theta = tf.zeros([chain_length,params.P])

		self.chain_length = chain_length
		self.burn = burn
		
	def update_state(self, step, state):
		self.p0[step] = state.p0
		self.vs[step] = state.vs
		self.sig2[step] = state.sig2
		self.z[step,:] = state.z
		self.theta[step,:] = state.theta

	def combine_chain(self, new_chain):
		self.p0 = tf.concatenate((self.p0, new_chain.p0),axis=0)
		self.vs = tf.concatenate((self.vs, new_chain.vs),axis=0)
		self.sig2 = tf.concatenate((self.sig2, new_chain.sig2),axis=0)
		self.z = tf.concatenate((self.z, new_chain.z),axis=0)
		self.theta = tf.concatenate((self.theta, new_chain.theta),axis=0)

	def burn_chain(self):
		self.p0 = tf.delete(self.p0,range(self.burn))
		self.vs = tf.delete(self.vs,range(self.burn))
		self.sig2 = tf.delete(se	lf.sig2,range(self.burn))
		self.z = tf.delete(self.z,range(self.burn),axis=0)
		self.theta = tf.delete(self.theta,range(self.burn),axis=0)

def inv_gamma(rng, alpha, beta):
	"""
	*** PDF for tf.linalg.inv_gamma:
	y = beta**(alpha) * x**(-alpha-1) * tf.exp(-beta/x) / sps.gamma(alpha)
	---
	"""
	sample = 1.0/rng.gamma(shape=alpha, scale=1.0/beta)  # FOR SAMPLING FROM IG, WE TAKE RECIPROCAL FROM GAMMA DISTRIBUTION
	return sample


def bernoulli(rng, p):
	u = rng.random()
	if u <= p:
		return 1
	else:
		return 0


def is_finite_and_real(x):
	if (tf.all(tf.isfinite(x))==False) or (tf.all(tf.isreal(x))==False):
		return False
	else:
		return True


def get_Arinv(Dr, N):
	"""
	DSS-i: The slab distribution is taken to be a positively truncated, uncorrrelated
	Gaussian distribution, because of which Artf.linalg.inv is taken to be the tf.eye matrix
	---
	"""
	Arinv = tf.eye(Dr.shape[1])

	return Arinv

def sample_theta(rng, data, params, state):

	theta = tf.zeros(params.P)

	s_z = tf.sum(state.z)

	if s_z > 0:

		Dr = data.D[:, state.z == 1]

		Arinv = get_Arinv(Dr, params.N)

		Einv = (Dr.T @ Dr + 1./state.vs * Arinv)
		E = tf.linalg.inv(Einv)

		mu = E @ Dr.T @ data.y


		lb = tf.zeros_like(mu) # lower bound
		ub = tf.ones_like(mu) * tf.inf # upper bound
		tmvn = TruncatedMVN(mu, state.sig2 * E, lb, ub) # Computing positively truncated multivariate normal distribution
		samples = tmvn.sample(1)[:,0]

		theta[state.z == 1] = samples

	state.theta = theta

	if is_finite_and_real(state.theta) == False:
		warnings.warn('\n ============ Not finite theta =========== \n')
		breakpoint()

	return state


def sample_sig2(rng, data, params, state):

	s_z = tf.sum(state.z)

	alpha = 0.
	beta = 0.

	if s_z > 0:

		Dr = data.D[:, state.z == 1]

		Arinv = get_Arinv(Dr, params.N)

		Einv = (Dr.T @ Dr + 1./state.vs * Arinv)
		E = tf.linalg.inv(Einv)

		mu = E @ Dr.T @ data.y

		alpha = params.a_sigma + params.N/2.0

		beta = params.b_sigma + ((data.y.T @ data.y) - (mu.T @ Einv @ mu))/2.0

	else:

		alpha = params.a_sigma + params.N/2.0

		beta = params.b_sigma + (data.y.T @ data.y)/2.0

	if alpha<0 or beta<0:
		warnings.warn('\n ============ Negative alpha/beta in sig2 =========== \n')
		breakpoint()

	state.sig2 = tf.linalg.inv_gamma(rng, alpha, beta)

	if is_finite_and_real(state.sig2) == False:
		warnings.warn('\n ============ Not finite sig2 =========== \n')
		breakpoint()

	return state


def sample_vs(rng, data, params, state):

	s_z = tf.sum(state.z)
	alpha = 0.
	beta = 0.

	if s_z > 0:
		Dr = data.D[:, state.z == 1]
		Arinv = get_Arinv(Dr, params.N)
		theta_r = state.theta[state.z == 1]
		alpha = params.a_v + s_z/2.0
		beta = params.b_v + (theta_r.T @ Arinv @ theta_r)/(2.0 * state.sig2)
	else:
		alpha = params.a_v
		beta = params.b_v

	if alpha<0 or beta<0:
		warnings.warn('\n ============ Negative alpha/beta in vs =========== \n')
		breakpoint()

	state.vs = tf.linalg.inv_gamma(rng, alpha, beta)

	if is_finite_and_real(state.vs) == False:
		warnings.warn('\n ============ Not finite vs =========== \n')
		breakpoint()

	return state



def sample_p0(rng, data, params, state):

	s_z = tf.sum(state.z)
	alpha = params.a_p + s_z
	beta = params.b_p + params.P - s_z

	if alpha<0 or beta<0:
		warnings.warn('\n ============ Negative alpha/beta in p0 =========== \n')
		breakpoint()

	state.p0 = rng.beta(alpha, beta)

	if is_finite_and_real(state.p0) == False:
		warnings.warn('\n ============ Not finite sig2 =========== \n')
		breakpoint()

	return state


def log_marginal_likelihood(rng, data, params, state):

	s_z = tf.sum(state.z)
	result = 0.

	if s_z > 0:
		t1 = (s_z/2.0) * log(state.vs)
		Dr = data.D[:, state.z == 1]
		Arinv = get_Arinv(Dr, params.N)
		t2 = 0.5 * log(tf.linalg.det(Arinv))
		M = tf.linalg.inv((Dr.T @ Dr) + (1./state.vs) * Arinv)
		t3 = 0.5 * log(tf.linalg.det(M))
		K = tf.eye(params.N) - Dr @ M @ Dr.T
		t4_base = params.b_sigma + 0.5 * data.y.T @ K @ data.y
		t4 = (params.a_sigma + 0.5*params.N) * tf.log(t4_base)
		result =  - t1 + t2 + t3 - t4

	else:
		t4_base = params.b_sigma + 0.5 * data.y.T @ data.y
		t4 = (params.a_sigma + 0.5*params.N) * tf.log(t4_base)
		result = - t4

	if is_finite_and_real(result) == False:
		warnings.warn('\n ============ Not finite log_marginal_likelihood =========== \n')
		breakpoint()

	return result



def sample_z(rng, data, params, state):

	idx = list(rng.permutation(params.P))

	for i in idx:
		state.z[i] = 0
		log_py_0 = log_marginal_likelihood(rng, data, params, state)
		state.z[i] = 1
		log_py_1 = log_marginal_likelihood(rng, data, params, state)
		ratio = tf.exp(log_py_0 - log_py_1)
		zeta_i = state.p0 / (state.p0 + ratio*(1.0-state.p0))
		state.z[i] = bernoulli(rng, zeta_i)

	if is_finite_and_real(state.z) == False:
		warnings.warn('\n ============ Not finite theta =========== \n')
		breakpoint()

	return state




def run_spike_slab(rng, data, params, chain_length, burn, chain_iter, verbose):
	"""
	Running the Monte-Carlo Markov chain sampling of the posterior distribution.

	_Input Arguments_

	-`rng` - Random number generator

	-`data` - Contains `D` (derivatives of feature library) and `y` (inertia terms and reaction forces)

	-`params` - Hyperparameters for Bayesian discovery: $a_{\nu}$, $b_{\nu}$, $a_{p}$, $b_{p}$, $a_{\sigma}$, $b_{\sigma}$

	-`chain_length`

	-`burn` - number of chain elements discarded as burn-in

	-`chain_iter` - id number of the parallel chain being constructed

	-`verbose` - toggle True or False for displaying chain progress

	_Output Arguments_

	- `chain` - Object of class `Chain`

	---

	"""
	#----------------------------------------------------------------------------------------
	# setup
	#----------------------------------------------------------------------------------------
	state = State(params)


	# add 5% random perturbation to initial values

	state.p0 = 0.1 * tf.random.uniform(0.95, 1.05);
	state.vs = 1. * tf.random.uniform(0.95, 1.05);
	state.sig2 = 1. * tf.random.uniform(0.95, 1.05);
	state.z[0]=1 * tf.random.uniform(0.95, 1.05);
	state.theta[0]=1. * tf.random.uniform(0.95, 1.05);

	chain = Chain(params, chain_length, burn)
	chain.update_state(0, state)

	#----------------------------------------------------------------------------------------
	# begin HSS
	#----------------------------------------------------------------------------------------
	if(verbose):
		print('\nBeginning chains (in parallel): \n')

# IS THIS GIBBS SAMPLING?
	for n in progressbar(range(1,chain_length), 'chain: ',show_progress_bar=verbose):

		state = sample_theta(rng, data, params, state)
		chain.update_state(n, state)
		state = sample_sig2(rng, data, params, state)
		chain.update_state(n, state)
		state = sample_vs(rng, data, params, state)
		chain.update_state(n, state)
		state = sample_p0(rng, data, params, state)
		chain.update_state(n, state)
		state = sample_z(rng, data, params, state)
		chain.update_state(n, state)

	chain.burn_chain()

	return chain

#
