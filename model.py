"""
This is the heart of the algorithm. Implements the objective function and mu
and sigma estimators for a Gaussian diffusion probabilistic model
"""

import numpy as np

# import theano
# import theano.tensor as T

# from blocks.bricks import application, Initializable, Random

import torch as T
from torch.nn import Module

import regression
import util
import config

class DiffusionModel(Module):
    def __init__(self,
            spatial_width,
            n_colors,
            trajectory_length=1000,
            n_temporal_basis=10,
            n_hidden_dense_lower=500,
            n_hidden_dense_lower_output=2,
            n_hidden_dense_upper=20,
            n_hidden_conv=20,
            n_layers_conv=4,
            n_layers_dense_lower=4,
            n_layers_dense_upper=2,
            n_t_per_minibatch=1,
            n_scales=1,
            step1_beta=0.001,
            uniform_noise = 0,
            ):
        """
        Implements the objective function and mu and sigma estimators for a Gaussian diffusion
        probabilistic model, as described in the paper:
            Deep Unsupervised Learning using Nonequilibrium Thermodynamics
            Jascha Sohl-Dickstein, Eric A. Weiss, Niru Maheswaranathan, Surya Ganguli
            International Conference on Machine Learning, 2015

        Parameters are as follow:
        spatial_width - Spatial_width of training images
        n_colors - Number of color channels in training data.
        trajectory_length - The number of time steps in the trajectory.
        n_temporal_basis - The number of temporal basis functions to capture time-step
            dependence of model.
        n_hidden_dense_lower - The number of hidden units in each layer of the dense network
            in the lower half of the MLP. Set to 0 to make a convolutional-only lower half.
        n_hidden_dense_lower_output - The number of outputs *per pixel* from the dense network
            in the lower half of the MLP. Total outputs are
            n_hidden_dense_lower_output*spatial_width**2.
        n_hidden_dense_upper - The number of hidden units per pixel in the upper half of the MLP.
        n_hidden_conv - The number of feature layers in the convolutional layers in the lower half
            of the MLP.
        n_layers_conv - How many convolutional layers to use in the lower half of the MLP.
        n_layers_dense_lower - How many dense layers to use in the lower half of the MLP.
        n_layers_dense_upper - How many dense layers to use in the upper half of the MLP.
        n_t_per_minibatch - When computing objective, how many random time-steps t to evaluate
            each minibatch at.
        step1_beta - The lower bound on the noise variance of the first diffusion step. This is
            the minimum variance of the learned model.
        uniform_noise - Add uniform noise between [-uniform_noise/2, uniform_noise/2] to the input.
        """
        super(DiffusionModel, self).__init__()

        self.n_t_per_minibatch = n_t_per_minibatch
        self.spatial_width = spatial_width
        self.n_colors = n_colors
        self.n_temporal_basis = n_temporal_basis
        self.trajectory_length = trajectory_length
        self.uniform_noise = uniform_noise

        self.mlp = regression.MLP_conv_dense(
            n_layers_conv, n_layers_dense_lower, n_layers_dense_upper,
            n_hidden_conv, n_hidden_dense_lower, n_hidden_dense_lower_output, n_hidden_dense_upper,
            spatial_width, n_colors, n_scales, n_temporal_basis)
        self.temporal_basis = self.generate_temporal_basis(trajectory_length, n_temporal_basis)
        self.beta_arr = self.generate_beta_arr(step1_beta)

    def generate_beta_arr(self, step1_beta):
        """
        Generate the noise covariances, beta_t, for the forward trajectory.
        """
        # lower bound on beta
        min_beta_val = 1e-6
        min_beta_values = T.ones((self.trajectory_length,))*min_beta_val
        min_beta_values[0] += step1_beta
        min_beta = min_beta_values.to(config.floatX)

        # (potentially learned) function for how beta changes with timestep
        # TODO add beta_perturb_coefficients to the parameters to be learned
        beta_perturb_coefficients_values = T.zeros((self.n_temporal_basis,))
        beta_perturb_coefficients = beta_perturb_coefficients_values.to(config.floatX)

        beta_perturb = T.matmul(self.temporal_basis.T, beta_perturb_coefficients)
        # baseline behavior of beta with time -- destroy a constant fraction
        # of the original data variance each time step
        # NOTE 2 below means a fraction ~1/T of the variance will be left at the end of the
        # trajectory
        beta_baseline = 1./T.linspace(self.trajectory_length, 2., self.trajectory_length)
        beta_baseline_offset = util.logit_np(beta_baseline).to(config.floatX)
        # and the actual beta_t, restricted to be between min_beta and 1-[small value]
        beta_arr = T.sigmoid(beta_perturb + beta_baseline_offset)
        beta_arr = min_beta + beta_arr * (1 - min_beta - 1e-5)
        beta_arr = beta_arr.reshape((self.trajectory_length, 1))
        return beta_arr


    def get_t_weights(self, t):
        """
        Generate vector of weights allowing selection of current timestep.
        (if t is not an integer, the weights will linearly interpolate)
        """
        n_seg = self.trajectory_length
        t_compare = T.arange(n_seg).reshape((1,n_seg))
        diff = abs(t - t_compare)
        t_weights, _ = T.max(T.cat(((-diff+1).reshape((n_seg,1)),
                           T.zeros((n_seg,1))), dim=1), 1)
        return t_weights.reshape((-1,1))


    def get_beta_forward(self, t):
        """
        Get the covariance of the forward diffusion process at timestep
        t.
        """
        t_weights = self.get_t_weights(t)
        return T.matmul(t_weights.T, self.beta_arr)


    def get_mu_sigma(self, X_noisy, t):
        """
        Generate mu and sigma for one step in the reverse trajectory,
        starting from a minibatch of images X_noisy, and at timestep t.
        """
        Z = self.mlp(X_noisy)
        mu_coeff, beta_coeff = self.temporal_readout(Z, t)
        # reverse variance is perturbation around forward variance
        beta_forward = self.get_beta_forward(t)
        # make impact of beta_coeff scaled appropriately with mu_coeff
        beta_coeff_scaled = beta_coeff / T.sqrt(T.tensor(self.trajectory_length))
        beta_reverse = T.sigmoid(beta_coeff_scaled + util.logit(beta_forward))
        # # reverse mean is decay towards mu_coeff
        # mu = (X_noisy - mu_coeff)*T.sqrt(1. - beta_reverse) + mu_coeff
        # reverse mean is a perturbation around the mean under forward
        # process


        # # DEBUG -- use these lines to test objective is 0 for isotropic Gaussian model
        # beta_reverse = beta_forward
        # mu_coeff = mu_coeff*0


        mu = X_noisy*T.sqrt(1. - beta_forward) + mu_coeff*T.sqrt(beta_forward)
        sigma = T.sqrt(beta_reverse)
        # mu.name = 'mu p'
        # sigma.name = 'sigma p'
        return mu, sigma


    def generate_forward_diffusion_sample(self, X_noiseless):
        """
        Corrupt a training image with t steps worth of Gaussian noise, and
        return the corrupted image, as well as the mean and covariance of the
        posterior q(x^{t-1}|x^t, x^0).
        """

        X_noiseless = X_noiseless.reshape(
            (-1, self.n_colors, self.spatial_width, self.spatial_width))

        n_images = X_noiseless.shape[0]
        # choose a timestep in [1, self.trajectory_length-1].
        # note the reverse process is fixed for the very
        # first timestep, so we skip it.
        # TODO for some reason random_integer is missing from the Blocks
        # theano random number generator.
        t = T.floor(T.empty(1,1).uniform_(1, self.trajectory_length).to(config.floatX))
        t_weights = self.get_t_weights(t)
        N = T.normal(0., 1.,
            (n_images, self.n_colors, self.spatial_width, self.spatial_width))

        # noise added this time step
        beta_forward = self.get_beta_forward(t)
        # decay in noise variance due to original signal this step
        alpha_forward = 1. - beta_forward
        # compute total decay in the fraction of the variance due to X_noiseless
        alpha_arr = 1. - self.beta_arr
        alpha_cum_forward_arr = T.cumprod(alpha_arr, 0).reshape((self.trajectory_length,1))
        alpha_cum_forward = T.matmul(t_weights.T, alpha_cum_forward_arr)
        # total fraction of the variance due to noise being mixed in
        beta_cumulative = 1. - alpha_cum_forward
        # total fraction of the variance due to noise being mixed in one step ago
        beta_cumulative_prior_step = 1. - alpha_cum_forward/alpha_forward

        # generate the corrupted training data
        X_uniformnoise = X_noiseless + (T.empty((n_images, self.n_colors, self.spatial_width, self.spatial_width), dtype=config.floatX).uniform_()-T.full((1,), 0.5,dtype=config.floatX))*T.full((1,), self.uniform_noise, dtype=config.floatX)
        X_noisy = X_uniformnoise*T.sqrt(alpha_cum_forward) + N*T.sqrt(1. - alpha_cum_forward)

        # compute the mean and covariance of the posterior distribution
        mu1_scl = T.sqrt(alpha_cum_forward / alpha_forward)
        mu2_scl = 1. / T.sqrt(alpha_forward)
        cov1 = 1. - alpha_cum_forward/alpha_forward
        cov2 = beta_forward / alpha_forward
        lam = 1./cov1 + 1./cov2
        mu = (
                X_uniformnoise * mu1_scl / cov1 +
                X_noisy * mu2_scl / cov2
            ) / lam
        sigma = T.sqrt(1./lam)
        sigma = sigma.reshape((1,1,1,1))

        # mu.name = 'mu q posterior'
        # sigma.name = 'sigma q posterior'
        # X_noisy.name = 'X_noisy'
        # t.name = 't'

        return X_noisy, t, mu, sigma


    def get_beta_full_trajectory(self):
        """
        Return the cumulative covariance from the entire forward trajectory.
        """
        alpha_arr = 1. - self.beta_arr
        beta_full_trajectory = 1. - T.exp(T.sum(T.log(alpha_arr)))
        return beta_full_trajectory


    def get_negL_bound(self, mu, sigma, mu_posterior, sigma_posterior):
        """
        Compute the lower bound on the log likelihood, as a function of mu and
        sigma from the reverse diffusion process, and the posterior mu and
        sigma from the forward diffusion process.

        Returns the difference between this bound and the log likelihood
        under a unit norm isotropic Gaussian. So this function returns how
        much better the diffusion model is than an isotropic Gaussian.
        """

        # the KL divergence between model transition and posterior from data
        KL = (  T.log(sigma) - T.log(sigma_posterior)
                + (sigma_posterior**2 + (mu_posterior-mu)**2)/(2*sigma**2)
                - 0.5)
        # conditional entropies H_q(x^T|x^0) and H_q(x^1|x^0)
        H_startpoint = (0.5*(1 + np.log(2.*np.pi))).astype(theano.config.floatX) + 0.5*T.log(self.beta_arr[0])
        H_endpoint = (0.5*(1 + np.log(2.*np.pi))).astype(theano.config.floatX) + 0.5*T.log(self.get_beta_full_trajectory())
        H_prior = (0.5*(1 + np.log(2.*np.pi))).astype(theano.config.floatX) + 0.5*T.log(1.)
        negL_bound = KL*self.trajectory_length + H_startpoint - H_endpoint + H_prior
        # the negL_bound if this was an isotropic Gaussian model of the data
        negL_gauss = (0.5*(1 + np.log(2.*np.pi))).astype(theano.config.floatX) + 0.5*T.log(1.)
        negL_diff = negL_bound - negL_gauss
        L_diff_bits = negL_diff / T.log(2.)
        L_diff_bits_avg = L_diff_bits.mean()*self.n_colors
        return L_diff_bits_avg


    def cost_single_t(self, X_noiseless):
        """
        Compute the lower bound on the log likelihood, given a training minibatch, for a single
        randomly chosen timestep.
        """
        X_noisy, t, mu_posterior, sigma_posterior = \
            self.generate_forward_diffusion_sample(X_noiseless)
        mu, sigma = self.get_mu_sigma(X_noisy, t)
        negL_bound = self.get_negL_bound(mu, sigma, mu_posterior, sigma_posterior)
        return negL_bound


    def internal_state(self, X_noiseless):
        """
        Return a bunch of the internal state, for monitoring purposes during optimization.
        """
        X_noisy, t, mu_posterior, sigma_posterior = \
            self.generate_forward_diffusion_sample(X_noiseless)
        mu, sigma = self.get_mu_sigma(X_noisy, t)
        mu_diff = mu-mu_posterior
        mu_diff.name = 'mu diff'
        logratio = T.log(sigma/sigma_posterior)
        logratio.name = 'log sigma ratio'
        return [mu_diff, logratio, mu, sigma, mu_posterior, sigma_posterior, X_noiseless, X_noisy]


    def cost(self, X_noiseless):
        """
        Compute the lower bound on the log likelihood, given a training minibatch.
        This will draw a single timestep and compute the cost for that timestep only.
        """
        cost = 0.
        for ii in range(self.n_t_per_minibatch):
            cost += self.cost_single_t(X_noiseless)
        return cost/self.n_t_per_minibatch


    def temporal_readout(self, Z, t):
        """
        Go from the top layer of the multilayer perceptron to coefficients for
        mu and sigma for each pixel.
        Z contains coefficients for spatial basis functions for each pixel for
        both mu and sigma.
        """
        n_images = Z.shape[0]
        t_weights = self.get_t_weights(t)
        Z = Z.reshape((n_images, self.spatial_width, self.spatial_width,
            self.n_colors, 2, self.n_temporal_basis))
        coeff_weights = T.matmul(self.temporal_basis, t_weights)
        concat_coeffs = T.matmul(Z, coeff_weights)
        mu_coeff = concat_coeffs[:,:,:,:,0].permute(0,3,1,2,4)
        beta_coeff = concat_coeffs[:,:,:,:,1].permute(0,3,1,2,4)
        return mu_coeff, beta_coeff


    def generate_temporal_basis(self, trajectory_length, n_basis):
        """
        Generate the bump basis functions for temporal readout of mu and sigma.
        """
        temporal_basis = T.zeros((trajectory_length, n_basis))
        xx = T.linspace(-1, 1, trajectory_length)
        x_centers = T.linspace(-1, 1, n_basis)
        width = (x_centers[1] - x_centers[0])/2.
        for ii in range(n_basis):
            temporal_basis[:,ii] = T.exp(-(xx-x_centers[ii])**2 / (2*width**2))
        temporal_basis /= T.sum(temporal_basis, axis=1).reshape((-1,1))
        temporal_basis = temporal_basis.T
        temporal_basis_theano = temporal_basis.to(config.floatX)
        return temporal_basis_theano
