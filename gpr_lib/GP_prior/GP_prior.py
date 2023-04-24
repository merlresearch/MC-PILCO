# Copyright (C) 2023 Alberto Dalla Libera
#
# SPDX-License-Identifier: MIT

"""
Author: Alberto Dalla Libera (alberto.dallalibera.1@gmail.com)
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from .. import Likelihood

# import scipy
# import quadprog
# from qpth.qp import QPFunction


class GP_prior(torch.nn.Module):
    """Superclass of GP models"""

    def __init__(
        self,
        active_dims,
        sigma_n_init=None,
        flg_train_sigma_n=True,
        name="",
        dtype=torch.float64,
        sigma_n_num=None,
        device=None,
    ):
        """Initialize the object Module objects. Values setted are:
        -noise information
        -data type
        -device
        -active dims"""
        super(GP_prior, self).__init__()
        # model name
        self.name = name
        # device
        self.dtype = dtype
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device
        # active dims
        if active_dims is None:
            self.active_dims = active_dims
        else:
            self.active_dims = torch.tensor(active_dims, requires_grad=False, device=device, dtype=torch.long)
        # sigma_noise measurement noise
        if sigma_n_init is None:
            self.GP_with_noise = False
        else:
            self.GP_with_noise = True
            self.sigma_n_log = torch.nn.Parameter(
                torch.tensor(np.log(sigma_n_init), dtype=self.dtype, device=self.device),
                requires_grad=flg_train_sigma_n,
            )
        # standard deviation of the rounding errors
        if sigma_n_num is not None:
            self.sigma_n_num = torch.tensor(sigma_n_num, dtype=self.dtype, device=self.device)
        else:
            self.sigma_n_num = torch.tensor(0.0, dtype=self.dtype, device=self.device)

    def to(self, dev):
        """Set the device and move the parameters"""
        super(GP_prior, self).to(dev)
        self.device = dev
        self.sigma_n_num = self.sigma_n_num.to(dev)

    def set_eval_mode(self):
        """Set the model in eval mode"""
        self.flg_trainable_list = []
        for p in self.parameters():
            self.flg_trainable_list.append(p.requires_grad)
            p.requires_grad = False

    def set_training_mode(self):
        """Set the model in training mode"""
        for i, p in enumerate(self.parameters()):
            p.requires_grad = self.flg_trainable_list[i]

    def get_sigma_n_2(self):
        """Returns the variance of the noise (measurement noise + rounding noise)"""
        return torch.exp(self.sigma_n_log) ** 2 + self.sigma_n_num**2

    def forward(self, X):
        """Returns the mean, variance, inverse of the variance and log_det.
        In order to compute the log_det efficiently this method computes
        the cholesky decomposition of the variance"""
        N = X.size()[0]
        if self.GP_with_noise:
            K_X = self.get_covariance(X, flg_noise=True)
        else:
            K_X = self.get_covariance(X)

        # inverse with LU decomposition (NO grad in logdet)
        # K_X_inv, LU = torch.gesv(torch.eye(K_X.size()[0], dtype=self.dtype, device=self.device), K_X)
        # log_det = torch.sum(torch.log(torch.diag(LU)**2))/2

        # inverse with cholesky
        U = torch.cholesky(K_X, upper=True)
        log_det = 2 * torch.sum(torch.log(torch.diag(U)))

        U_inv = torch.inverse(U)
        K_X_inv = torch.matmul(U_inv, U_inv.transpose(0, 1))

        # K_X_inv = torch.cholesky_inverse(U, upper=True)

        m_X = self.get_mean(X)
        return m_X, K_X, K_X_inv, log_det

    def get_mean(self, X):
        """Returns the prior mean in X"""
        raise NotImplementedError()

    def get_covariance(self, X1, X2=None, flg_noise=False):
        """Returns the covariance betweeen the input locations in X1 X2.
        If X2 is None X2 is assumed to be equal to X1"""
        raise NotImplementedError()

    def get_diag_covariance(self, X, flg_noise=False):
        """Returns the diagonal elements of the covariance betweeen the input locations in X1 X2"""
        raise NotImplementedError()

    def get_alpha(self, X, Y):
        """Returns the alpha vector, the vector of the optimal coefficient"""
        m_X, _, K_X_inv, _ = self(X)
        alpha = torch.matmul(K_X_inv, Y - m_X)

        return alpha, m_X, K_X_inv

    def get_estimate_from_alpha(self, X, X_test, alpha, m_X, K_X_inv=None, Y_test=None):
        """Performs estimation on X_test given the alpha coefficient associated to X.
        If K_X_inv is given the method returns also the confidence intervals (variance of the gaussian)
        If Y_test is given the method prints the MSE"""
        # get covariance and mean
        K_X_test_X = self.get_covariance(X_test, X)
        m_X_test = self.get_mean(X_test)
        # get the estimate
        Y_hat = m_X_test + torch.matmul(K_X_test_X, alpha)
        # print the MSE is Y_test is given
        if not (Y_test is None):
            print("MSE:", torch.sum((Y_test - Y_hat) ** 2) / Y_test.size()[0])
        # if K_X_inv is given compute the confidence intervals
        if K_X_inv is not None:
            num_test = X_test.size()[0]
            var = self.get_diag_covariance(X_test) - torch.sum(torch.matmul(K_X_test_X, K_X_inv) * (K_X_test_X), dim=1)
            return Y_hat, var
        else:
            return Y_hat

    def get_estimate(self, X, Y, X_test, Y_test=None, flg_return_K_X_inv=False):
        """Returns the estimate in X_test given the observations X Y.
        The function returns also:
        -a vector containing the sigma squared confidence intervals
        -the vector of the coefficient
        -the K_X inverse in case required through flg_return_K_X_inv"""
        # get the coefficent and the mean
        alpha, m_X, K_X_inv = self.get_alpha(X, Y)
        # get the estimate and the confidence intervals
        Y_hat, var = self.get_estimate_from_alpha(X, X_test, alpha, m_X, K_X_inv=K_X_inv, Y_test=Y_test)
        # return the opportune values
        if flg_return_K_X_inv:
            return Y_hat, var, alpha, m_X, K_X_inv
        else:
            return Y_hat, var, alpha

    def print_model(self):
        """Print the model parameters"""
        print(self.name + " parameters:")
        for par_name, par_value in self.named_parameters():
            print("-", par_name, ":", par_value.data)

    def fit_model(
        self,
        trainloader=None,
        optimizer=None,
        criterion=None,
        N_epoch=1,
        N_epoch_print=1,  # flg_time=False,
        f_saving_model=None,
        f_print=None,
    ):
        """Performs the optimization of the model. The function considered is the forward fucntion,
        i.e. the inputs of the criterion are [m_X, K_X, K_X_inv, log_det]"""
        # print initial parametes and initial estimates
        print("\nInitial parameters:")
        self.print_model()
        # iterate over the training data for N_epochs
        t_start = time.time()
        for epoch in range(0, N_epoch):
            # initialize loss grad and counter
            running_loss = 0.0
            N_btc = 0
            optimizer.zero_grad()
            # iterate over the training set
            # print('\nEPOCH:', epoch)
            for i, data in enumerate(trainloader, 0):
                # get the training data
                inputs, labels = data
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                out_GP_priors = self(inputs)
                loss = criterion(out_GP_priors, labels)
                loss.backward(retain_graph=False)
                optimizer.step()
                # update the running loss
                running_loss = running_loss + loss.item()
                N_btc = N_btc + 1
            # print statistics and save the model
            if epoch % N_epoch_print == 0:
                print("\nEPOCH:", epoch)
                self.print_model()
                print("Running loss:", running_loss / N_btc)
                t_stop = time.time()
                print("Time elapsed:", t_stop - t_start)
                t_start = time.time()
                if f_saving_model is not None:
                    f_saving_model(epoch)
                if f_print is not None:
                    f_print()
        # print the final parameters
        print("\nFinal parameters:")
        self.print_model()

    def get_SOD(self, X, Y, threshold, flg_permutation=False):
        """
        Returns the SOD points with an online procedure
        SOD: most importants subset of data
        """
        print("\nSelection of the inducing inputs...")
        # get number of samples
        num_samples = X.shape[0]
        # init the set of inducing inputs with the first sample
        SOD = X[0:1, :]
        inducing_inputs_indices = [0]
        # get a permuation of the inputs
        perm_indices = torch.arange(1, num_samples)
        if flg_permutation:
            perm_indices = perm_indices[torch.randperm(num_samples - 1)]
        # iterate all the samples
        for sample_index in perm_indices:
            # get the estimate
            _, var, _ = self.get_estimate(
                X[inducing_inputs_indices, :], Y[inducing_inputs_indices, :], X[sample_index : sample_index + 1, :]
            )
            if torch.sqrt(var) > threshold:
                SOD = torch.cat([SOD, X[sample_index : sample_index + 1, :]], 0)
                inducing_inputs_indices.append(sample_index)
        print("Shape inducing inputs selected:", SOD.shape)
        return inducing_inputs_indices


class Combine_GP(GP_prior):
    """Class that extend GP_prior and provide common utilities to multiple kernel operations"""

    def __init__(self, *gp_priors_obj):
        """Initialize the multiple kernel object"""
        # initialize a new GP object
        super(Combine_GP, self).__init__(
            active_dims=None,
            sigma_n_num=gp_priors_obj[0].sigma_n_num,
            dtype=gp_priors_obj[0].dtype,
            device=gp_priors_obj[0].device,
        )
        # build a list with all the models
        self.gp_list = torch.nn.ModuleList(gp_priors_obj)
        # check the noise flag
        GP_with_noise = False
        for gp in self.gp_list:
            GP_with_noise = GP_with_noise or gp.GP_with_noise
        self.GP_with_noise = GP_with_noise

    def to(self, dev):
        super(Combine_GP, self).to(dev)
        self.device = dev
        for gp in self.gp_list:
            gp.to(dev)

    def print_model(self):
        for gp in self.gp_list:
            gp.print_model()

    def get_sigma_n_2(self):
        """Iterate over all the models in the list and returns the noise variance"""
        sigma_n_2 = torch.zeros(1, dtype=self.dtype, device=self.device)
        for gp in self.gp_list:
            if gp.GP_with_noise:
                sigma_n_2 += gp.get_sigma_n_2()
        return sigma_n_2


class Sum_Independent_GP(Combine_GP):
    """Class that sum independent GP_priors objects"""

    def __init__(self, *gp_priors_obj):
        """Initialize the gp list"""
        super(Sum_Independent_GP, self).__init__(*gp_priors_obj)

    def get_mean(self, X):
        """Returns the sum of the means of the GP in gp_list"""
        N = X.size()[0]
        mean = torch.zeros(N, 1, dtype=self.dtype, device=self.device)
        for gp in self.gp_list:
            mean += gp.get_mean(X)
            return mean

    def get_covariance(self, X1, X2=None, flg_noise=False):
        """Returns the sum of the covariances of the gp_list"""
        # get dimensions
        N1 = X1.size()[0]
        if X2 is None:
            N2 = N1
        else:
            N2 = X2.size()[0]
        # #initialize the covariance
        # cov = torch.zeros(N1,N2, dtype=self.dtype, device=self.device)
        # #sum all the covariances
        # for gp in self.gp_list:
        #     cov += gp.get_covariance(X1,X2, flg_noise=False)
        # f_gp = lambda gp : (gp.get_covariance(X1,X2,flg_noise=False)).unsqueeze(0)
        # cov = torch.sum(torch.cat(list(map(f_gp, self.gp_list)),0),0)
        cov = torch.sum(
            torch.cat([(gp.get_covariance(X1, X2, flg_noise=False)).unsqueeze(0) for gp in self.gp_list], 0), 0
        )
        # add the noise
        if flg_noise & self.GP_with_noise:
            cov += self.get_sigma_n_2() * torch.eye(N1, dtype=self.dtype, device=self.device)
        return cov

    def get_diag_covariance(self, X, flg_noise=False):
        """Returns the sum of the diagonals of the covariances in the gp list"""
        # initialize the vector
        diag = torch.zeros(X.size()[0], dtype=self.dtype, device=self.device)
        # iterate in the list and sum the diagonals
        for gp in self.gp_list:
            diag += gp.get_diag_covariance(X, flg_noise=False)
        # add the noise
        if flg_noise & self.GP_with_noise:
            diag += self.get_sigma_n_2()
        return diag


class Multiply_GP_prior(Combine_GP):
    """Class that generate a GP_prior multiplying GP_priors objects"""

    def __init__(self, *gp_priors_obj):
        """Initilize the GP list"""
        super(Multiply_GP_prior, self).__init__(*gp_priors_obj)

    def get_mean(self, X):
        """Returns the product of the means of the GP in gp_list"""
        # initilize the mean vector
        N = X.size()[0]
        mean = torch.ones(N, 1, dtype=self.dtype, device=self.device)
        # multiply all the means
        for gp in self.gp_list:
            mean = mean * gp.get_mean(X)
        return mean

    def get_covariance(self, X1, X2=None, flg_noise=False):
        """Returns the element-wise product of the covariances og the GP in gp_list"""
        # get size
        N1 = X1.size()[0]
        if X2 is None:
            N2 = N1
        else:
            N2 = X2.size()[0]
        # #initilize the covariance
        # cov = torch.ones(N1,N2, dtype=self.dtype, device=self.device)
        # #multiply all the covariances
        # for gp in self.gp_list:
        #     cov *=gp.get_covariance(X1,X2, flg_noise=False)
        # f_gp = lambda gp : (gp.get_covariance(X1,X2,flg_noise=False)).unsqueeze(0)
        # cov = torch.prod(torch.cat(list(map(f_gp, self.gp_list)),0),0)
        cov = torch.prod(
            torch.cat([(gp.get_covariance(X1, X2, flg_noise=False)).unsqueeze(0) for gp in self.gp_list], 0), 0
        )
        # add the noise
        if flg_noise & self.GP_with_noise:
            cov += self.get_sigma_n_2() * torch.eye(N1, dtype=self.dtype, device=self.device)
        return cov

    def get_diag_covariance(self, X, flg_noise=False):
        """Returns the product of the diagonals vector relative to the covariance of the GP in gp_list"""
        # initilize the diagona
        N = X.size()[0]
        diag = torch.ones(N, dtype=self.dtype, device=self.device)
        # multiply all the diagonals
        for gp in self.gp_list:
            diag *= gp.get_diag_covariance(X, flg_noise=False)
        # add the nosie
        if flg_noise & self.GP_with_noise:
            diag += self.get_sigma_n_2()
        return diag


def Scale_GP_prior(
    GP_prior_class,
    GP_prior_par_dict,
    f_scale,
    active_dims_f_scale,
    pos_par_f_init=None,
    flg_train_pos_par_f=True,
    free_par_f_init=None,
    flg_train_free_par_f=True,
    additional_par_f_list=[],
):
    """Funciton that returns a GP_prior scaled. This class implement the following model:
    y(x) = a(x)f(x) + e, where f(x) is a GP and a(x) a deterministic function.
    The function a() can be parametrize respect to a set of trainable prameters.
    This class retuns an instance of a new class defined inside"""

    # define the new class
    class Scaled_GP(GP_prior_class):
        """Class that extends the GP_prior_class with the scaling parameters"""

        def __init__(
            self,
            GP_prior_par_dict,
            f_scale,
            active_dims_f_scale,
            pos_par_f,
            flg_train_pos_par_f,
            free_par_f,
            flg_train_free_par_f,
            additional_par_f_list,
        ):
            # initialize the object of the superclass
            super(Scaled_GP, self).__init__(**GP_prior_par_dict)
            # save the scaling info
            self.f_scale = f_scale
            self.active_dims_f_scale = active_dims_f_scale
            self.additional_par_f_list = additional_par_f_list
            if pos_par_f_init:
                self.flg_pos_par = True
                self.pos_par_f_log = torch.nn.Parameter(
                    torch.tensor(np.log(pos_par_f_init), dtype=self.dtype, device=self.device),
                    requires_grad=flg_train_pos_par_f,
                )
            else:
                self.flg_pos_par = False
                self.pos_par_f_log = None
            if free_par_f_init:
                self.flg_free_par = True
                self.free_par_f = torch.nn.Parameter(
                    torch.tensor(free_par_f_init, dtype=self.dtype, device=self.device),
                    requires_grad=flg_train_free_par_f,
                )
            else:
                self.flg_free_par = False
                self.free_par_f = None

        def get_scaling(self, X):
            """Returns the scaling funciton evaluated in X"""
            if self.flg_pos_par:
                pos_par = torch.exp(self.pos_par_f_log)
            else:
                pos_par = None
            return self.f_scale(
                X[:, self.active_dims_f_scale], pos_par, self.free_par_f, *self.additional_par_f_list
            ).reshape(-1)

        def get_mean(self, X):
            """Calls the get_mean of the superclass and apply the scaling"""
            # get the supercalss mean
            return self.get_scaling(X) * super(Scaled_GP, self).get_mean(X)

        def get_covariance(self, X1, X2=None, flg_noise=False):
            """Calls the get covariance of the superclass and apply the scaling"""
            # get the scaling functions
            a_X1 = self.get_scaling(X1)
            # if required evaluate the scaling function in X2 and get the covariance
            if X2 is None:
                # print(super(Scaled_GP, self).get_covariance(X1, X2, flg_noise=False))
                K = a_X1 * super(Scaled_GP, self).get_covariance(X1, X2, flg_noise=False) * (a_X1.transpose(0, 1))
            else:
                a_X2 = self.get_scaling(X2)
                K = a_X1 * super(Scaled_GP, self).get_covariance(X1, X2, flg_noise=False) * (a_X2.transpose(0, 1))
            # if required add the noise and return the covariance
            if flg_noise & self.GP_with_noise:
                # print('noise')
                N = K.size()[0]
                return K + self.get_sigma_n_2() * torch.eye(N, dtype=self.dtype, device=self.device)
            else:
                return K

        def get_diag_covariance(self, X, flg_noise=False):
            """Calls the get_diag_covariance of the superclass and apply the scaling"""
            # evaluate the scaling function in X1
            a_X = self.get_scaling(X1)
            diag = a_X**2 * super(Scaled_GP, self).get_diag_covariance(X, flg_noise=False)
            # if required add the noise and return the covariance
            if flg_noise & self.GP_with_noise:
                return diag + self.get_sigma_n_2()
            else:
                return diag

    # return an object of the new class
    return Scaled_GP(
        GP_prior_par_dict,
        f_scale,
        active_dims_f_scale,
        pos_par_f_init,
        flg_train_pos_par_f,
        free_par_f_init,
        flg_train_free_par_f,
        additional_par_f_list,
    )
