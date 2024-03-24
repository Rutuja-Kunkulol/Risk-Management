#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy as scp
from scipy.sparse.linalg import spsolve
from scipy import sparse
from scipy.sparse.linalg import splu
import matplotlib.pyplot as plt
from matplotlib import cm
from time import time
import scipy.stats as ss
from scipy.fftpack import ifft
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.optimize import fsolve
import scipy as scp
from scipy import sparse
from scipy.sparse.linalg import splu
from time import time
from functools import partial
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds, LinearConstraint
from scipy.integrate import quad
import scipy.special as scps
from math import factorial
from statsmodels.tools.numdiff import approx_hess
import pandas as pd
from scipy.linalg import norm, solve_triangular
from scipy.linalg.lapack import get_lapack_funcs
from scipy.linalg.misc import LinAlgError


# In[2]:


def cf_normal(u, mu=1, sig=2):
    """
    Characteristic function of a Normal random variable
    """
    return np.exp(1j * u * mu - 0.5 * u**2 * sig**2)


# In[3]:


def cf_gamma(u, a=1, b=2):
    """
    Characteristic function of a Gamma random variable
    - shape: a
    - scale: b
    """
    return (1 - b * u * 1j) ** (-a)


# In[4]:


def cf_poisson(u, lam=1):
    """
    Characteristic function of a Poisson random variable
    - rate: lam
    """
    return np.exp(lam * (np.exp(1j * u) - 1))


# In[5]:


def cf_mert(u, t=1, mu=1, sig=2, lam=0.8, muJ=0, sigJ=0.5):
    """
    Characteristic function of a Merton random variable at time t
    mu: drift
    sig: diffusion coefficient
    lam: jump activity
    muJ: jump mean size
    sigJ: jump size standard deviation
    """
    return np.exp(
        t * (1j * u * mu - 0.5 * u**2 * sig**2 + lam * (np.exp(1j * u * muJ - 0.5 * u**2 * sigJ**2) - 1))
    )


# In[6]:


def cf_VG(u, t=1, mu=0, theta=-0.1, sigma=0.2, kappa=0.1):
    """
    Characteristic function of a Variance Gamma random variable at time t
    mu: additional drift
    theta: Brownian motion drift
    sigma: Brownian motion diffusion
    kappa: Gamma process variance
    """
    return np.exp(t * (1j * mu * u - np.log(1 - 1j * theta * kappa * u + 0.5 * kappa * sigma**2 * u**2) / kappa))


# In[7]:


def cf_NIG(u, t=1, mu=0, theta=-0.1, sigma=0.2, kappa=0.1):
    """
    Characteristic function of a Normal Inverse Gaussian random variable at time t
    mu: additional drift
    theta: Brownian motion drift
    sigma: Brownian motion diffusion
    kappa: Inverse Gaussian process variance
    """
    return np.exp(
        t * (1j * mu * u + 1 / kappa - np.sqrt(1 - 2j * theta * kappa * u + kappa * sigma**2 * u**2) / kappa)
    )


# In[8]:


def cf_Heston(u, t, v0, mu, kappa, theta, sigma, rho):
    """
    Heston characteristic function as proposed in the original paper of Heston (1993)
    """
    xi = kappa - sigma * rho * u * 1j
    d = np.sqrt(xi**2 + sigma**2 * (u**2 + 1j * u))
    g1 = (xi + d) / (xi - d)
    cf = np.exp(
        1j * u * mu * t
        + (kappa * theta) / (sigma**2) * ((xi + d) * t - 2 * np.log((1 - g1 * np.exp(d * t)) / (1 - g1)))
        + (v0 / sigma**2) * (xi + d) * (1 - np.exp(d * t)) / (1 - g1 * np.exp(d * t))
    )
    return cf


# In[9]:


def cf_Heston_good(u, t, v0, mu, kappa, theta, sigma, rho):
    """
    Heston characteristic function as proposed by Schoutens (2004)
    """
    xi = kappa - sigma * rho * u * 1j
    d = np.sqrt(xi**2 + sigma**2 * (u**2 + 1j * u))
    g1 = (xi + d) / (xi - d)
    g2 = 1 / g1
    cf = np.exp(
        1j * u * mu * t
        + (kappa * theta) / (sigma**2) * ((xi - d) * t - 2 * np.log((1 - g2 * np.exp(-d * t)) / (1 - g2)))
        + (v0 / sigma**2) * (xi - d) * (1 - np.exp(-d * t)) / (1 - g2 * np.exp(-d * t))
    )
    return cf


# In[10]:


def fft_Lewis(K, S0, r, T, cf, interp="cubic"):
    """
    K = vector of strike
    S = spot price scalar
    cf = characteristic function
    interp can be cubic or linear
    """
    N = 2**15  # FFT more efficient for N power of 2
    B = 500  # integration limit
    dx = B / N
    x = np.arange(N) * dx  # the final value B is excluded

    weight = np.arange(N)  # Simpson weights
    weight = 3 + (-1) ** (weight + 1)
    weight[0] = 1
    weight[N - 1] = 1

    dk = 2 * np.pi / B
    b = N * dk / 2
    ks = -b + dk * np.arange(N)

    integrand = np.exp(-1j * b * np.arange(N) * dx) * cf(x - 0.5j) * 1 / (x**2 + 0.25) * weight * dx / 3
    integral_value = np.real(ifft(integrand) * N)

    if interp == "linear":
        spline_lin = interp1d(ks, integral_value, kind="linear")
        prices = S0 - np.sqrt(S0 * K) * np.exp(-r * T) / np.pi * spline_lin(np.log(S0 / K))
    elif interp == "cubic":
        spline_cub = interp1d(ks, integral_value, kind="cubic")
        prices = S0 - np.sqrt(S0 * K) * np.exp(-r * T) / np.pi * spline_cub(np.log(S0 / K))
    return prices


# In[11]:


def IV_from_Lewis(K, S0, T, r, cf, disp=False):
    """Implied Volatility from the Lewis formula
    K = strike; S0 = spot stock; T = time to maturity; r = interest rate
    cf = characteristic function"""
    k = np.log(S0 / K)

    def obj_fun(sig):
        integrand = (
            lambda u: np.real(
                np.exp(u * k * 1j)
                * (cf(u - 0.5j) - np.exp(1j * u * r * T + 0.5 * r * T) * np.exp(-0.5 * T * (u**2 + 0.25) * sig**2))
            )
            * 1
            / (u**2 + 0.25)
        )
        int_value = quad(integrand, 1e-15, 2000, limit=2000, full_output=1)[0]
        return int_value

    X0 = [0.2, 1, 2, 4, 0.0001]  # set of initial guess points
    for x0 in X0:
        x, _, solved, msg = fsolve(
            obj_fun,
            [
                x0,
            ],
            full_output=True,
            xtol=1e-4,
        )
        if solved == 1:
            return x[0]
    if disp is True:
        print("Strike", K, msg)
    return -1


# In[12]:


class Heston_pricer:
    """
    Class to price the options with the Heston model by:
    - Fourier-inversion.
    - Monte Carlo.
    """

    def __init__(self, Option_info, Process_info):
        """
        Process_info:  of type VG_process. It contains the interest rate r
        and the VG parameters (sigma, theta, kappa)

        Option_info:  of type Option_param. It contains (S0,K,T) i.e. current price,
        strike, maturity in years
        """
        self.r = Process_info.mu  # interest rate
        self.sigma = Process_info.sigma  # Heston parameter
        self.theta = Process_info.theta  # Heston parameter
        self.kappa = Process_info.kappa  # Heston parameter
        self.rho = Process_info.rho  # Heston parameter

        self.S0 = Option_info.S0  # current price
        self.v0 = Option_info.v0  # spot variance
        self.K = Option_info.K  # strike
        self.T = Option_info.T  # maturity in years

        self.exercise = Option_info.exercise
        self.payoff = Option_info.payoff

    def payoff_f(self, S):
        if self.payoff == "call":
            Payoff = np.maximum(S - self.K, 0)
        elif self.payoff == "put":
            Payoff = np.maximum(self.K - S, 0)
        return Payoff

    def MC(self, N, paths, Err=False, Time=False):
        """
        Heston Monte Carlo
        N = time steps
        paths = number of simulated paths
        Err = return Standard Error if True
        Time = return execution time if True
        """
        t_init = time()

        S_T, _ = Heston_paths(
            N=N,
            paths=paths,
            T=self.T,
            S0=self.S0,
            v0=self.v0,
            mu=self.r,
            rho=self.rho,
            kappa=self.kappa,
            theta=self.theta,
            sigma=self.sigma,
        )
        S_T = S_T.reshape((paths, 1))
        DiscountedPayoff = np.exp(-self.r * self.T) * self.payoff_f(S_T)
        V = scp.mean(DiscountedPayoff, axis=0)
        std_err = ss.sem(DiscountedPayoff)

        if Err is True:
            if Time is True:
                elapsed = time() - t_init
                return V, std_err, elapsed
            else:
                return V, std_err
        else:
            if Time is True:
                elapsed = time() - t_init
                return V, elapsed
            else:
                return V

    def Fourier_inversion(self):
        """
        Price obtained by inversion of the characteristic function
        """
        k = np.log(self.K / self.S0)  # log moneyness
        cf_H_b_good = partial(
            cf_Heston_good,
            t=self.T,
            v0=self.v0,
            mu=self.r,
            theta=self.theta,
            sigma=self.sigma,
            kappa=self.kappa,
            rho=self.rho,
        )

        limit_max = 2000  # right limit in the integration

        if self.payoff == "call":
            call = self.S0 * Q1(k, cf_H_b_good, limit_max) - self.K * np.exp(-self.r * self.T) * Q2(
                k, cf_H_b_good, limit_max
            )
            return call
        elif self.payoff == "put":
            put = self.K * np.exp(-self.r * self.T) * (1 - Q2(k, cf_H_b_good, limit_max)) - self.S0 * (
                1 - Q1(k, cf_H_b_good, limit_max)
            )
            return put
        else:
            raise ValueError("invalid type. Set 'call' or 'put'")

    def FFT(self, K):
        """
        FFT method. It returns a vector of prices.
        K is an array of strikes
        """
        K = np.array(K)
        cf_H_b_good = partial(
            cf_Heston_good,
            t=self.T,
            v0=self.v0,
            mu=self.r,
            theta=self.theta,
            sigma=self.sigma,
            kappa=self.kappa,
            rho=self.rho,
        )

        if self.payoff == "call":
            return fft_Lewis(K, self.S0, self.r, self.T, cf_H_b_good, interp="cubic")
        elif self.payoff == "put":  # put-call parity
            return (
                fft_Lewis(K, self.S0, self.r, self.T, cf_H_b_good, interp="cubic")
                - self.S0
                + K * np.exp(-self.r * self.T)
            )
        else:
            raise ValueError("invalid type. Set 'call' or 'put'")

    def IV_Lewis(self):
        """Implied Volatility from the Lewis formula"""

        cf_H_b_good = partial(
            cf_Heston_good,
            t=self.T,
            v0=self.v0,
            mu=self.r,
            theta=self.theta,
            sigma=self.sigma,
            kappa=self.kappa,
            rho=self.rho,
        )
        if self.payoff == "call":
            return IV_from_Lewis(self.K, self.S0, self.T, self.r, cf_H_b_good)
        elif self.payoff == "put":
            raise NotImplementedError
        else:
            raise ValueError("invalid type. Set 'call' or 'put'")


# In[13]:


class Kalman_regression:
    """Kalman Filter algorithm for the linear regression beta estimation.
    Alpha is assumed constant.

    INPUT:
    X = predictor variable. ndarray, Series or DataFrame.
    Y = response variable.
    alpha0 = constant alpha. The regression intercept.
    beta0 = initial beta.
    var_eta = variance of process error
    var_eps = variance of measurement error
    P0 = initial covariance of beta
    """

    def __init__(self, X, Y, alpha0=None, beta0=None, var_eta=None, var_eps=None, P0=10):
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.var_eta = var_eta
        self.var_eps = var_eps
        self.P0 = P0
        self.X = np.asarray(X)
        self.Y = np.asarray(Y)
        self.loglikelihood = None
        self.R2_pre_fit = None
        self.R2_post_fit = None

        self.betas = None
        self.Ps = None

        if (self.alpha0 is None) or (self.beta0 is None) or (self.var_eps is None):
            self.alpha0, self.beta0, self.var_eps = self.get_OLS_params()
            print("alpha0, beta0 and var_eps initialized by OLS")

    ####################  enforce X and Y to be numpy arrays ######################
    #    @property
    #    def X(self):
    #        return self._X
    #    @X.setter
    #    def X(self, value):
    #        if not isinstance(value, np.ndarray):
    #            raise TypeError('X must be a numpy array')
    #        self._X = value
    #
    #    @property
    #    def Y(self):
    #        return self._Y
    #    @Y.setter
    #    def Y(self, value):
    #        if not isinstance(value, np.ndarray):
    #            raise TypeError('Y must be a numpy array')
    #        self._Y = value
    ###############################################################################

    def get_OLS_params(self):
        """Returns the OLS alpha, beta and sigma^2 (variance of epsilon)
        Y = alpha + beta * X + epsilon
        """
        beta, alpha, _, _, _ = ss.linregress(self.X, self.Y)
        resid = self.Y - beta * self.X - alpha
        sig2 = resid.var(ddof=2)
        return alpha, beta, sig2

    def set_OLS_params(self):
        self.alpha0, self.beta0, self.var_eps = self.get_OLS_params()

    def run(self, X=None, Y=None, var_eta=None, var_eps=None):
        """
        Run the Kalman Filter
        """

        if (X is None) and (Y is None):
            X = self.X
            Y = self.Y

        X = np.asarray(X)
        Y = np.asarray(Y)

        N = len(X)
        if len(Y) != N:
            raise ValueError("Y and X must have same length")

        if var_eta is not None:
            self.var_eta = var_eta
        if var_eps is not None:
            self.var_eps = var_eps
        if self.var_eta is None:
            raise ValueError("var_eta is None")

        betas = np.zeros_like(X)
        Ps = np.zeros_like(X)
        res_pre = np.zeros_like(X)  # pre-fit residuals

        Y = Y - self.alpha0  # re-define Y
        P = self.P0
        beta = self.beta0

        log_2pi = np.log(2 * np.pi)
        loglikelihood = 0

        for k in range(N):
            # Prediction
            beta_p = beta  # predicted beta
            P_p = P + self.var_eta  # predicted P

            # ausiliary variables
            r = Y[k] - beta_p * X[k]
            S = P_p * X[k] ** 2 + self.var_eps
            KG = X[k] * P_p / S  # Kalman gain

            # Update
            beta = beta_p + KG * r
            P = P_p * (1 - KG * X[k])

            loglikelihood += 0.5 * (-log_2pi - np.log(S) - (r**2 / S))

            betas[k] = beta
            Ps[k] = P
            res_pre[k] = r

        res_post = Y - X * betas  # post fit residuals
        sqr_err = Y - np.mean(Y)
        R2_pre = 1 - (res_pre @ res_pre) / (sqr_err @ sqr_err)
        R2_post = 1 - (res_post @ res_post) / (sqr_err @ sqr_err)

        self.loglikelihood = loglikelihood
        self.R2_post_fit = R2_post
        self.R2_pre_fit = R2_pre

        self.betas = betas
        self.Ps = Ps

    def calibrate_MLE(self):
        """Returns the result of the MLE calibration for the Beta Kalman filter,
        using the L-BFGS-B method.
        The calibrated parameters are var_eta and var_eps.
        X, Y          = Series, array, or DataFrame for the regression
        alpha_tr      = initial alpha
        beta_tr       = initial beta
        var_eps_ols   = initial guess for the errors
        """

        def minus_likelihood(c):
            """Function to minimize in order to calibrate the kalman parameters:
            var_eta and var_eps."""
            self.var_eps = c[0]
            self.var_eta = c[1]
            self.run()
            return -1 * self.loglikelihood

        result = minimize(
            minus_likelihood,
            x0=[self.var_eps, self.var_eps],
            method="L-BFGS-B",
            bounds=[[1e-15, None], [1e-15, None]],
            tol=1e-6,
        )

        if result.success is True:
            self.beta0 = self.betas[-1]
            self.P0 = self.Ps[-1]
            self.var_eps = result.x[0]
            self.var_eta = result.x[1]


# In[14]:


class Merton_pricer:
    """
    Closed Formula.
    Monte Carlo.
    Finite-difference PIDE: Explicit-implicit scheme

        0 = dV/dt + (r -(1/2)sig^2 -m) dV/dx + (1/2)sig^2 d^V/dx^2
                 + \int[ V(x+y) nu(dy) ] -(r+lam)V
    """

    def __init__(self, Option_info, Process_info):
        """
        Process_info:  of type Merton_process. It contains (r, sig, lam, muJ, sigJ) i.e.
        interest rate, diffusion coefficient, jump activity and jump distribution params

        Option_info:  of type Option_param. It contains (S0,K,T) i.e. current price,
        strike, maturity in years
        """
        self.r = Process_info.r  # interest rate
        self.sig = Process_info.sig  # diffusion coefficient
        self.lam = Process_info.lam  # jump activity
        self.muJ = Process_info.muJ  # jump mean
        self.sigJ = Process_info.sigJ  # jump std
        self.exp_RV = Process_info.exp_RV  # function to generate exponential Merton Random Variables

        self.S0 = Option_info.S0  # current price
        self.K = Option_info.K  # strike
        self.T = Option_info.T  # maturity in years

        self.price = 0
        self.S_vec = None
        self.price_vec = None
        self.mesh = None
        self.exercise = Option_info.exercise
        self.payoff = Option_info.payoff

    def payoff_f(self, S):
        if self.payoff == "call":
            Payoff = np.maximum(S - self.K, 0)
        elif self.payoff == "put":
            Payoff = np.maximum(self.K - S, 0)
        return Payoff

    def closed_formula(self):
        """
        Merton closed formula.
        """

        m = self.lam * (np.exp(self.muJ + (self.sigJ**2) / 2) - 1)  # coefficient m
        lam2 = self.lam * np.exp(self.muJ + (self.sigJ**2) / 2)

        tot = 0
        for i in range(18):
            tot += (np.exp(-lam2 * self.T) * (lam2 * self.T) ** i / factorial(i)) * BS_pricer.BlackScholes(
                self.payoff,
                self.S0,
                self.K,
                self.T,
                self.r - m + i * (self.muJ + 0.5 * self.sigJ**2) / self.T,
                np.sqrt(self.sig**2 + (i * self.sigJ**2) / self.T),
            )
        return tot

    def Fourier_inversion(self):
        """
        Price obtained by inversion of the characteristic function
        """
        k = np.log(self.K / self.S0)  # log moneyness
        m = self.lam * (np.exp(self.muJ + (self.sigJ**2) / 2) - 1)  # coefficient m
        cf_Mert = partial(
            cf_mert,
            t=self.T,
            mu=(self.r - 0.5 * self.sig**2 - m),
            sig=self.sig,
            lam=self.lam,
            muJ=self.muJ,
            sigJ=self.sigJ,
        )

        if self.payoff == "call":
            call = self.S0 * Q1(k, cf_Mert, np.inf) - self.K * np.exp(-self.r * self.T) * Q2(
                k, cf_Mert, np.inf
            )  # pricing function
            return call
        elif self.payoff == "put":
            put = self.K * np.exp(-self.r * self.T) * (1 - Q2(k, cf_Mert, np.inf)) - self.S0 * (
                1 - Q1(k, cf_Mert, np.inf)
            )  # pricing function
            return put
        else:
            raise ValueError("invalid type. Set 'call' or 'put'")

    def FFT(self, K):
        """
        FFT method. It returns a vector of prices.
        K is an array of strikes
        """
        K = np.array(K)
        m = self.lam * (np.exp(self.muJ + (self.sigJ**2) / 2) - 1)  # coefficient m
        cf_Mert = partial(
            cf_mert,
            t=self.T,
            mu=(self.r - 0.5 * self.sig**2 - m),
            sig=self.sig,
            lam=self.lam,
            muJ=self.muJ,
            sigJ=self.sigJ,
        )

        if self.payoff == "call":
            return fft_Lewis(K, self.S0, self.r, self.T, cf_Mert, interp="cubic")
        elif self.payoff == "put":  # put-call parity
            return (
                fft_Lewis(K, self.S0, self.r, self.T, cf_Mert, interp="cubic") - self.S0 + K * np.exp(-self.r * self.T)
            )
        else:
            raise ValueError("invalid type. Set 'call' or 'put'")

    def IV_Lewis(self):
        """Implied Volatility from the Lewis formula"""

        m = self.lam * (np.exp(self.muJ + (self.sigJ**2) / 2) - 1)  # coefficient m
        cf_Mert = partial(
            cf_mert,
            t=self.T,
            mu=(self.r - 0.5 * self.sig**2 - m),
            sig=self.sig,
            lam=self.lam,
            muJ=self.muJ,
            sigJ=self.sigJ,
        )

        if self.payoff == "call":
            return IV_from_Lewis(self.K, self.S0, self.T, self.r, cf_Mert)
        elif self.payoff == "put":
            raise NotImplementedError
        else:
            raise ValueError("invalid type. Set 'call' or 'put'")

    def MC(self, N, Err=False, Time=False):
        """
        Merton Monte Carlo
        Err = return Standard Error if True
        Time = return execution time if True
        """
        t_init = time()

        S_T = self.exp_RV(self.S0, self.T, N)
        V = scp.mean(np.exp(-self.r * self.T) * self.payoff_f(S_T), axis=0)

        if Err is True:
            if Time is True:
                elapsed = time() - t_init
                return V, ss.sem(np.exp(-self.r * self.T) * self.payoff_f(S_T)), elapsed
            else:
                return V, ss.sem(np.exp(-self.r * self.T) * self.payoff_f(S_T))
        else:
            if Time is True:
                elapsed = time() - t_init
                return V, elapsed
            else:
                return V

    def PIDE_price(self, steps, Time=False):
        """
        steps = tuple with number of space steps and time steps
        payoff = "call" or "put"
        exercise = "European" or "American"
        Time = Boolean. Execution time.
        """
        t_init = time()

        Nspace = steps[0]
        Ntime = steps[1]

        S_max = 6 * float(self.K)
        S_min = float(self.K) / 6
        x_max = np.log(S_max)
        x_min = np.log(S_min)

        dev_X = np.sqrt(self.lam * self.sigJ**2 + self.lam * self.muJ**2)

        dx = (x_max - x_min) / (Nspace - 1)
        extraP = int(np.floor(5 * dev_X / dx))  # extra points beyond the B.C.
        x = np.linspace(x_min - extraP * dx, x_max + extraP * dx, Nspace + 2 * extraP)  # space discretization
        t, dt = np.linspace(0, self.T, Ntime, retstep=True)  # time discretization

        Payoff = self.payoff_f(np.exp(x))
        offset = np.zeros(Nspace - 2)
        V = np.zeros((Nspace + 2 * extraP, Ntime))  # grid initialization

        if self.payoff == "call":
            V[:, -1] = Payoff  # terminal conditions
            V[-extraP - 1 :, :] = np.exp(x[-extraP - 1 :]).reshape(extraP + 1, 1) * np.ones(
                (extraP + 1, Ntime)
            ) - self.K * np.exp(-self.r * t[::-1]) * np.ones(
                (extraP + 1, Ntime)
            )  # boundary condition
            V[: extraP + 1, :] = 0
        else:
            V[:, -1] = Payoff
            V[-extraP - 1 :, :] = 0
            V[: extraP + 1, :] = self.K * np.exp(-self.r * t[::-1]) * np.ones((extraP + 1, Ntime))

        cdf = ss.norm.cdf(
            [np.linspace(-(extraP + 1 + 0.5) * dx, (extraP + 1 + 0.5) * dx, 2 * (extraP + 2))],
            loc=self.muJ,
            scale=self.sigJ,
        )[0]
        nu = self.lam * (cdf[1:] - cdf[:-1])

        lam_appr = sum(nu)
        m_appr = np.array([np.exp(i * dx) - 1 for i in range(-(extraP + 1), extraP + 2)]) @ nu

        sig2 = self.sig**2
        dxx = dx**2
        a = (dt / 2) * ((self.r - m_appr - 0.5 * sig2) / dx - sig2 / dxx)
        b = 1 + dt * (sig2 / dxx + self.r + lam_appr)
        c = -(dt / 2) * ((self.r - m_appr - 0.5 * sig2) / dx + sig2 / dxx)

        D = sparse.diags([a, b, c], [-1, 0, 1], shape=(Nspace - 2, Nspace - 2)).tocsc()
        DD = splu(D)
        if self.exercise == "European":
            for i in range(Ntime - 2, -1, -1):
                offset[0] = a * V[extraP, i]
                offset[-1] = c * V[-1 - extraP, i]
                V_jump = V[extraP + 1 : -extraP - 1, i + 1] + dt * signal.convolve(
                    V[:, i + 1], nu[::-1], mode="valid", method="fft"
                )
                V[extraP + 1 : -extraP - 1, i] = DD.solve(V_jump - offset)
        elif self.exercise == "American":
            for i in range(Ntime - 2, -1, -1):
                offset[0] = a * V[extraP, i]
                offset[-1] = c * V[-1 - extraP, i]
                V_jump = V[extraP + 1 : -extraP - 1, i + 1] + dt * signal.convolve(
                    V[:, i + 1], nu[::-1], mode="valid", method="fft"
                )
                V[extraP + 1 : -extraP - 1, i] = np.maximum(DD.solve(V_jump - offset), Payoff[extraP + 1 : -extraP - 1])

        X0 = np.log(self.S0)  # current log-price
        self.S_vec = np.exp(x[extraP + 1 : -extraP - 1])  # vector of S
        self.price = np.interp(X0, x, V[:, 0])
        self.price_vec = V[extraP + 1 : -extraP - 1, 0]
        self.mesh = V[extraP + 1 : -extraP - 1, :]

        if Time is True:
            elapsed = time() - t_init
            return self.price, elapsed
        else:
            return self.price

    def plot(self, axis=None):
        if type(self.S_vec) != np.ndarray or type(self.price_vec) != np.ndarray:
            self.PIDE_price((5000, 4000))

        plt.plot(self.S_vec, self.payoff_f(self.S_vec), color="blue", label="Payoff")
        plt.plot(self.S_vec, self.price_vec, color="red", label="Merton curve")
        if type(axis) == list:
            plt.axis(axis)
        plt.xlabel("S")
        plt.ylabel("price")
        plt.title("Merton price")
        plt.legend(loc="upper left")
        plt.show()

    def mesh_plt(self):
        if type(self.S_vec) != np.ndarray or type(self.mesh) != np.ndarray:
            self.PDE_price((7000, 5000))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        X, Y = np.meshgrid(np.linspace(0, self.T, self.mesh.shape[1]), self.S_vec)
        ax.plot_surface(Y, X, self.mesh, cmap=cm.ocean)
        ax.set_title("Merton price surface")
        ax.set_xlabel("S")
        ax.set_ylabel("t")
        ax.set_zlabel("V")
        ax.view_init(30, -100)  # this function rotates the 3d plot
        plt.show()


# In[15]:


class NIG_pricer:
    """
    Closed Formula.
    Monte Carlo.
    Finite-difference PIDE: Explicit-implicit scheme, with Brownian approximation

        0 = dV/dt + (r -(1/2)sig^2 -w) dV/dx + (1/2)sig^2 d^V/dx^2
                 + \int[ V(x+y) nu(dy) ] -(r+lam)V
    """

    def __init__(self, Option_info, Process_info):
        """
        Process_info:  of type NIG_process. It contains the interest rate r
        and the NIG parameters (sigma, theta, kappa)

        Option_info:  of type Option_param.
        It contains (S0,K,T) i.e. current price, strike, maturity in years
        """
        self.r = Process_info.r  # interest rate
        self.sigma = Process_info.sigma  # NIG parameter
        self.theta = Process_info.theta  # NIG parameter
        self.kappa = Process_info.kappa  # NIG parameter
        self.exp_RV = Process_info.exp_RV  # function to generate exponential NIG Random Variables

        self.S0 = Option_info.S0  # current price
        self.K = Option_info.K  # strike
        self.T = Option_info.T  # maturity in years

        self.price = 0
        self.S_vec = None
        self.price_vec = None
        self.mesh = None
        self.exercise = Option_info.exercise
        self.payoff = Option_info.payoff

    def payoff_f(self, S):
        if self.payoff == "call":
            Payoff = np.maximum(S - self.K, 0)
        elif self.payoff == "put":
            Payoff = np.maximum(self.K - S, 0)
        return Payoff

    def Fourier_inversion(self):
        """
        Price obtained by inversion of the characteristic function
        """
        k = np.log(self.K / self.S0)  # log moneyness
        w = (
            1 - np.sqrt(1 - 2 * self.theta * self.kappa - self.kappa * self.sigma**2)
        ) / self.kappa  # martingale correction

        cf_NIG_b = partial(
            cf_NIG,
            t=self.T,
            mu=(self.r - w),
            theta=self.theta,
            sigma=self.sigma,
            kappa=self.kappa,
        )

        if self.payoff == "call":
            call = self.S0 * Q1(k, cf_NIG_b, np.inf) - self.K * np.exp(-self.r * self.T) * Q2(
                k, cf_NIG_b, np.inf
            )  # pricing function
            return call
        elif self.payoff == "put":
            put = self.K * np.exp(-self.r * self.T) * (1 - Q2(k, cf_NIG_b, np.inf)) - self.S0 * (
                1 - Q1(k, cf_NIG_b, np.inf)
            )  # pricing function
            return put
        else:
            raise ValueError("invalid type. Set 'call' or 'put'")

    def MC(self, N, Err=False, Time=False):
        """
        NIG Monte Carlo
        Err = return Standard Error if True
        Time = return execution time if True
        """
        t_init = time()

        S_T = self.exp_RV(self.S0, self.T, N)
        V = scp.mean(np.exp(-self.r * self.T) * self.payoff_f(S_T))

        if Err is True:
            if Time is True:
                elapsed = time() - t_init
                return V, ss.sem(np.exp(-self.r * self.T) * self.payoff_f(S_T)), elapsed
            else:
                return V, ss.sem(np.exp(-self.r * self.T) * self.payoff_f(S_T))
        else:
            if Time is True:
                elapsed = time() - t_init
                return V, elapsed
            else:
                return V

    def NIG_measure(self, x):
        A = self.theta / (self.sigma**2)
        B = np.sqrt(self.theta**2 + self.sigma**2 / self.kappa) / self.sigma**2
        C = np.sqrt(self.theta**2 + self.sigma**2 / self.kappa) / (np.pi * self.sigma * np.sqrt(self.kappa))
        return C / np.abs(x) * np.exp(A * (x)) * scps.kv(1, B * np.abs(x))

    def PIDE_price(self, steps, Time=False):
        """
        steps = tuple with number of space steps and time steps
        payoff = "call" or "put"
        exercise = "European" or "American"
        Time = Boolean. Execution time.
        """
        t_init = time()

        Nspace = steps[0]
        Ntime = steps[1]

        S_max = 2000 * float(self.K)
        S_min = float(self.K) / 2000
        x_max = np.log(S_max)
        x_min = np.log(S_min)

        dev_X = np.sqrt(self.sigma**2 + self.theta**2 * self.kappa)  # std dev NIG process

        dx = (x_max - x_min) / (Nspace - 1)
        extraP = int(np.floor(7 * dev_X / dx))  # extra points beyond the B.C.
        x = np.linspace(x_min - extraP * dx, x_max + extraP * dx, Nspace + 2 * extraP)  # space discretization
        t, dt = np.linspace(0, self.T, Ntime, retstep=True)  # time discretization

        Payoff = self.payoff_f(np.exp(x))
        offset = np.zeros(Nspace - 2)
        V = np.zeros((Nspace + 2 * extraP, Ntime))  # grid initialization

        if self.payoff == "call":
            V[:, -1] = Payoff  # terminal conditions
            V[-extraP - 1 :, :] = np.exp(x[-extraP - 1 :]).reshape(extraP + 1, 1) * np.ones(
                (extraP + 1, Ntime)
            ) - self.K * np.exp(-self.r * t[::-1]) * np.ones(
                (extraP + 1, Ntime)
            )  # boundary condition
            V[: extraP + 1, :] = 0
        else:
            V[:, -1] = Payoff
            V[-extraP - 1 :, :] = 0
            V[: extraP + 1, :] = self.K * np.exp(-self.r * t[::-1]) * np.ones((extraP + 1, Ntime))

        eps = 1.5 * dx  # the cutoff near 0
        lam = (
            quad(self.NIG_measure, -(extraP + 1.5) * dx, -eps)[0] + quad(self.NIG_measure, eps, (extraP + 1.5) * dx)[0]
        )  # approximated intensity

        def int_w(y):
            return (np.exp(y) - 1) * self.NIG_measure(y)

        def int_s(y):
            return y**2 * self.NIG_measure(y)

        w = quad(int_w, -(extraP + 1.5) * dx, -eps)[0] + quad(int_w, eps, (extraP + 1.5) * dx)[0]  # is the approx of w
        sig2 = quad(int_s, -eps, eps, points=0)[0]  # the small jumps variance

        dxx = dx * dx
        a = (dt / 2) * ((self.r - w - 0.5 * sig2) / dx - sig2 / dxx)
        b = 1 + dt * (sig2 / dxx + self.r + lam)
        c = -(dt / 2) * ((self.r - w - 0.5 * sig2) / dx + sig2 / dxx)
        D = sparse.diags([a, b, c], [-1, 0, 1], shape=(Nspace - 2, Nspace - 2)).tocsc()
        DD = splu(D)

        nu = np.zeros(2 * extraP + 3)  # Lévy measure vector
        x_med = extraP + 1  # middle point in nu vector
        x_nu = np.linspace(-(extraP + 1 + 0.5) * dx, (extraP + 1 + 0.5) * dx, 2 * (extraP + 2))  # integration domain
        for i in range(len(nu)):
            if (i == x_med) or (i == x_med - 1) or (i == x_med + 1):
                continue
            nu[i] = quad(self.NIG_measure, x_nu[i], x_nu[i + 1])[0]

        if self.exercise == "European":
            # Backward iteration
            for i in range(Ntime - 2, -1, -1):
                offset[0] = a * V[extraP, i]
                offset[-1] = c * V[-1 - extraP, i]
                V_jump = V[extraP + 1 : -extraP - 1, i + 1] + dt * signal.convolve(
                    V[:, i + 1], nu[::-1], mode="valid", method="auto"
                )
                V[extraP + 1 : -extraP - 1, i] = DD.solve(V_jump - offset)
        elif self.exercise == "American":
            for i in range(Ntime - 2, -1, -1):
                offset[0] = a * V[extraP, i]
                offset[-1] = c * V[-1 - extraP, i]
                V_jump = V[extraP + 1 : -extraP - 1, i + 1] + dt * signal.convolve(
                    V[:, i + 1], nu[::-1], mode="valid", method="auto"
                )
                V[extraP + 1 : -extraP - 1, i] = np.maximum(DD.solve(V_jump - offset), Payoff[extraP + 1 : -extraP - 1])

        X0 = np.log(self.S0)  # current log-price
        self.S_vec = np.exp(x[extraP + 1 : -extraP - 1])  # vector of S
        self.price = np.interp(X0, x, V[:, 0])
        self.price_vec = V[extraP + 1 : -extraP - 1, 0]
        self.mesh = V[extraP + 1 : -extraP - 1, :]

        if Time is True:
            elapsed = time() - t_init
            return self.price, elapsed
        else:
            return self.price

    def plot(self, axis=None):
        if type(self.S_vec) != np.ndarray or type(self.price_vec) != np.ndarray:
            self.PIDE_price((5000, 4000))

        plt.plot(self.S_vec, self.payoff_f(self.S_vec), color="blue", label="Payoff")
        plt.plot(self.S_vec, self.price_vec, color="red", label="NIG curve")
        if type(axis) == list:
            plt.axis(axis)
        plt.xlabel("S")
        plt.ylabel("price")
        plt.title("NIG price")
        plt.legend(loc="best")
        plt.show()

    def mesh_plt(self):
        if type(self.S_vec) != np.ndarray or type(self.mesh) != np.ndarray:
            self.PDE_price((7000, 5000))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        X, Y = np.meshgrid(np.linspace(0, self.T, self.mesh.shape[1]), self.S_vec)
        ax.plot_surface(Y, X, self.mesh, cmap=cm.ocean)
        ax.set_title("NIG price surface")
        ax.set_xlabel("S")
        ax.set_ylabel("t")
        ax.set_zlabel("V")
        ax.view_init(30, -100)  # this function rotates the 3d plot
        plt.show()


# In[16]:



class Option_param:
    """
    Option class wants the option parameters:
    S0 = current stock price
    K = Strike price
    T = time to maturity
    v0 = (optional) spot variance
    exercise = European or American
    """

    def __init__(self, S0=15, K=15, T=1, v0=0.04, payoff="call", exercise="European"):
        self.S0 = S0
        self.v0 = v0
        self.K = K
        self.T = T

        if exercise == "European" or exercise == "American":
            self.exercise = exercise
        else:
            raise ValueError("invalid type. Set 'European' or 'American'")

        if payoff == "call" or payoff == "put":
            self.payoff = payoff
        else:
            raise ValueError("invalid type. Set 'call' or 'put'")


# In[17]:





def optimal_weights(MU, COV, Rf=0, w_max=1, desired_mean=None, desired_std=None):
    """
    Compute the optimal weights for a portfolio containing a risk free asset and stocks.
    MU = vector of mean
    COV = covariance matrix
    Rf = risk free return
    w_max = maximum weight bound for the stock portfolio
    desired_mean = desired mean of the portfolio
    desired_std = desired standard deviation of the portfolio
    """

    if (desired_mean is not None) and (desired_std is not None):
        raise ValueError("One among desired_mean and desired_std must be None")
    if ((desired_mean is not None) or (desired_std is not None)) and Rf == 0:
        raise ValueError("We just optimize the Sharpe ratio, no computation of efficient frontier")

    N = len(MU)
    bounds = Bounds(0, w_max)
    linear_constraint = LinearConstraint(np.ones(N, dtype=int), 1, 1)
    weights = np.ones(N)
    x0 = weights / np.sum(weights)  # initial guess

    def sharpe_fun(w):
        return -(MU @ w - Rf) / np.sqrt(w.T @ COV @ w)

    res = minimize(
        sharpe_fun,
        x0=x0,
        method="trust-constr",
        constraints=linear_constraint,
        bounds=bounds,
    )
    print(res.message + "\n")
    w_sr = res.x
    std_stock_portf = np.sqrt(w_sr @ COV @ w_sr)
    mean_stock_portf = MU @ w_sr
    stock_port_results = {
        "Sharpe Ratio": -sharpe_fun(w_sr),
        "stock weights": w_sr.round(4),
        "stock portfolio": {
            "std": std_stock_portf.round(6),
            "mean": mean_stock_portf.round(6),
        },
    }

    if (desired_mean is None) and (desired_std is None):
        return stock_port_results

    elif (desired_mean is None) and (desired_std is not None):
        w_stock = desired_std / std_stock_portf
        if desired_std > std_stock_portf:
            print(
                "The risk you take is higher than the tangency portfolio risk \
                ==> SHORT POSTION"
            )
        tot_port_mean = Rf + w_stock * (mean_stock_portf - Rf)
        return {
            **stock_port_results,
            "Bond + Stock weights": {
                "Bond": (1 - w_stock).round(4),
                "Stock": w_stock.round(4),
            },
            "Total portfolio": {"std": desired_std, "mean": tot_port_mean.round(6)},
        }

    elif (desired_mean is not None) and (desired_std is None):
        w_stock = (desired_mean - Rf) / (mean_stock_portf - Rf)
        if desired_mean > mean_stock_portf:
            print(
                "The return you want is higher than the tangency portfolio return \
                    ==> SHORT POSTION"
            )
        tot_port_std = w_stock * std_stock_portf
        return {
            **stock_port_results,
            "Bond + Stock weights": {
                "Bond": (1 - w_stock).round(4),
                "Stock": w_stock.round(4),
            },
            "Total portfolio": {"std": tot_port_std.round(6), "mean": desired_mean},
        }


# In[18]:


def calc_Q1(k, cf, right_lim):
    """
    P(X<k) - Probability to be in the money under the stock numeraire.
    cf: characteristic function
    right_lim: right limit of integration
    """

    def integrand(u):
        return np.real((np.exp(-u * k * 1j) / (u * 1j)) * cf(u - 1j) / cf(-1.0000000000001j))

    return 1 / 2 + 1 / np.pi * quad(integrand, 1e-15, right_lim, limit=2000)[0]


def calc_Q2(k, cf, right_lim):
    """
    P(X<k) - Probability to be in the money under the money market numeraire
    cf: characteristic function
    right_lim: right limit of integration
    """

    def integrand(u):
        return np.real(np.exp(-u * k * 1j) / (u * 1j) * cf(u))

    return 1 / 2 + 1 / np.pi * quad(integrand, 1e-15, right_lim, limit=2000)[0]


# In[19]:


def Gil_Pelaez_pdf(x, cf, right_lim):
    """
    Gil Pelaez formula for the inversion of the characteristic function
    INPUT
    - x: is a number
    - right_lim: is the right extreme of integration
    - cf: is the characteristic function
    OUTPUT
    - the value of the density at x.
    """

    def integrand(u):
        return np.real(np.exp(-u * x * 1j) * cf(u))

    return 1 / np.pi * quad(integrand, 1e-15, right_lim)[0]


def Heston_pdf(i, t, v0, mu, theta, sigma, kappa, rho):
    """
    Heston density by Fourier inversion.
    """
    cf_H_b_good = partial(
        cf_Heston_good,
        t=t,
        v0=v0,
        mu=mu,
        theta=theta,
        sigma=sigma,
        kappa=kappa,
        rho=rho,
    )
    return Gil_Pelaez_pdf(i, cf_H_b_good, np.inf)


def VG_pdf(x, T, c, theta, sigma, kappa):
    """
    Variance Gamma density function
    """
    return (
        2
        * np.exp(theta * (x - c) / sigma**2)
        / (kappa ** (T / kappa) * np.sqrt(2 * np.pi) * sigma * scps.gamma(T / kappa))
        * ((x - c) ** 2 / (2 * sigma**2 / kappa + theta**2)) ** (T / (2 * kappa) - 1 / 4)
        * scps.kv(
            T / kappa - 1 / 2,
            sigma ** (-2) * np.sqrt((x - c) ** 2 * (2 * sigma**2 / kappa + theta**2)),
        )
    )


def Merton_pdf(x, T, mu, sig, lam, muJ, sigJ):
    """
    Merton density function
    """
    tot = 0
    for k in range(20):
        tot += (
            (lam * T) ** k
            * np.exp(-((x - mu * T - k * muJ) ** 2) / (2 * (T * sig**2 + k * sigJ**2)))
            / (factorial(k) * np.sqrt(2 * np.pi * (sig**2 * T + k * sigJ**2)))
        )
    return np.exp(-lam * T) * tot


def NIG_pdf(x, T, c, theta, sigma, kappa):
    """
    Merton density function
    """
    A = theta / (sigma**2)
    B = np.sqrt(theta**2 + sigma**2 / kappa) / sigma**2
    C = T / np.pi * np.exp(T / kappa) * np.sqrt(theta**2 / (kappa * sigma**2) + 1 / kappa**2)
    return (
        C
        * np.exp(A * (x - c * T))
        * scps.kv(1, B * np.sqrt((x - c * T) ** 2 + T**2 * sigma**2 / kappa))
        / np.sqrt((x - c * T) ** 2 + T**2 * sigma**2 / kappa)
    )


# In[20]:


class Diffusion_process:
    """
    Class for the diffusion process:
    r = risk free constant rate
    sig = constant diffusion coefficient
    mu = constant drift
    """

    def __init__(self, r=0.1, sig=0.2, mu=0.1):
        self.r = r
        self.mu = mu
        if sig <= 0:
            raise ValueError("sig must be positive")
        else:
            self.sig = sig

    def exp_RV(self, S0, T, N):
        W = ss.norm.rvs((self.r - 0.5 * self.sig**2) * T, np.sqrt(T) * self.sig, N)
        S_T = S0 * np.exp(W)
        return S_T.reshape((N, 1))


class Merton_process:
    """
    Class for the Merton process:
    r = risk free constant rate
    sig = constant diffusion coefficient
    lam = jump activity
    muJ = jump mean
    sigJ = jump standard deviation
    """

    def __init__(self, r=0.1, sig=0.2, lam=0.8, muJ=0, sigJ=0.5):
        self.r = r
        self.lam = lam
        self.muJ = muJ
        if sig < 0 or sigJ < 0:
            raise ValueError("sig and sigJ must be positive")
        else:
            self.sig = sig
            self.sigJ = sigJ

        # moments
        self.var = self.sig**2 + self.lam * self.sigJ**2 + self.lam * self.muJ**2
        self.skew = self.lam * (3 * self.sigJ**2 * self.muJ + self.muJ**3) / self.var ** (1.5)
        self.kurt = self.lam * (3 * self.sigJ**3 + 6 * self.sigJ**2 * self.muJ**2 + self.muJ**4) / self.var**2

    def exp_RV(self, S0, T, N):
        m = self.lam * (np.exp(self.muJ + (self.sigJ**2) / 2) - 1)  # coefficient m
        W = ss.norm.rvs(0, 1, N)  # The normal RV vector
        P = ss.poisson.rvs(self.lam * T, size=N)  # Poisson random vector (number of jumps)
        Jumps = np.asarray([ss.norm.rvs(self.muJ, self.sigJ, ind).sum() for ind in P])  # Jumps vector
        S_T = S0 * np.exp(
            (self.r - 0.5 * self.sig**2 - m) * T + np.sqrt(T) * self.sig * W + Jumps
        )  # Martingale exponential Merton
        return S_T.reshape((N, 1))


class VG_process:
    """
    Class for the Variance Gamma process:
    r = risk free constant rate
    Using the representation of Brownian subordination, the parameters are:
        theta = drift of the Brownian motion
        sigma = standard deviation of the Brownian motion
        kappa = variance of the of the Gamma process
    """

    def __init__(self, r=0.1, sigma=0.2, theta=-0.1, kappa=0.1):
        self.r = r
        self.c = self.r
        self.theta = theta
        self.kappa = kappa
        if sigma < 0:
            raise ValueError("sigma must be positive")
        else:
            self.sigma = sigma

        # moments
        self.mean = self.c + self.theta
        self.var = self.sigma**2 + self.theta**2 * self.kappa
        self.skew = (2 * self.theta**3 * self.kappa**2 + 3 * self.sigma**2 * self.theta * self.kappa) / (
            self.var ** (1.5)
        )
        self.kurt = (
            3 * self.sigma**4 * self.kappa
            + 12 * self.sigma**2 * self.theta**2 * self.kappa**2
            + 6 * self.theta**4 * self.kappa**3
        ) / (self.var**2)

    def exp_RV(self, S0, T, N):
        w = -np.log(1 - self.theta * self.kappa - self.kappa / 2 * self.sigma**2) / self.kappa  # coefficient w
        rho = 1 / self.kappa
        G = ss.gamma(rho * T).rvs(N) / rho  # The gamma RV
        Norm = ss.norm.rvs(0, 1, N)  # The normal RV
        VG = self.theta * G + self.sigma * np.sqrt(G) * Norm  # VG process at final time G
        S_T = S0 * np.exp((self.r - w) * T + VG)  # Martingale exponential VG
        return S_T.reshape((N, 1))

    def path(self, T=1, N=10000, paths=1):
        """
        Creates Variance Gamma paths
        N = number of time points (time steps are N-1)
        paths = number of generated paths
        """
        dt = T / (N - 1)  # time interval
        X0 = np.zeros((paths, 1))
        G = ss.gamma(dt / self.kappa, scale=self.kappa).rvs(size=(paths, N - 1))  # The gamma RV
        Norm = ss.norm.rvs(loc=0, scale=1, size=(paths, N - 1))  # The normal RV
        increments = self.c * dt + self.theta * G + self.sigma * np.sqrt(G) * Norm
        X = np.concatenate((X0, increments), axis=1).cumsum(1)
        return X

    def fit_from_data(self, data, dt=1, method="Nelder-Mead"):
        """
        Fit the 4 parameters of the VG process using MM (method of moments),
        Nelder-Mead, L-BFGS-B.

        data (array): datapoints
        dt (float):     is the increment time

        Returns (c, theta, sigma, kappa)
        """
        X = data
        sigma_mm = np.std(X) / np.sqrt(dt)
        kappa_mm = dt * ss.kurtosis(X) / 3
        theta_mm = np.sqrt(dt) * ss.skew(X) * sigma_mm / (3 * kappa_mm)
        c_mm = np.mean(X) / dt - theta_mm

        def log_likely(x, data, T):
            return (-1) * np.sum(np.log(VG_pdf(data, T, x[0], x[1], x[2], x[3])))

        if method == "L-BFGS-B":
            if theta_mm < 0:
                result = minimize(
                    log_likely,
                    x0=[c_mm, theta_mm, sigma_mm, kappa_mm],
                    method="L-BFGS-B",
                    args=(X, dt),
                    tol=1e-8,
                    bounds=[[-0.5, 0.5], [-0.6, -1e-15], [1e-15, 1], [1e-15, 2]],
                )
            else:
                result = minimize(
                    log_likely,
                    x0=[c_mm, theta_mm, sigma_mm, kappa_mm],
                    method="L-BFGS-B",
                    args=(X, dt),
                    tol=1e-8,
                    bounds=[[-0.5, 0.5], [1e-15, 0.6], [1e-15, 1], [1e-15, 2]],
                )
            print(result.message)
        elif method == "Nelder-Mead":
            result = minimize(
                log_likely,
                x0=[c_mm, theta_mm, sigma_mm, kappa_mm],
                method="Nelder-Mead",
                args=(X, dt),
                options={"disp": False, "maxfev": 3000},
                tol=1e-8,
            )
            print(result.message)
        elif "MM":
            self.c, self.theta, self.sigma, self.kappa = (
                c_mm,
                theta_mm,
                sigma_mm,
                kappa_mm,
            )
            return
        self.c, self.theta, self.sigma, self.kappa = result.x


class Heston_process:
    """
    Class for the Heston process:
    r = risk free constant rate
    rho = correlation between stock noise and variance noise
    theta = long term mean of the variance process
    sigma = volatility coefficient of the variance process
    kappa = mean reversion coefficient for the variance process
    """

    def __init__(self, mu=0.1, rho=0, sigma=0.2, theta=-0.1, kappa=0.1):
        self.mu = mu
        if np.abs(rho) > 1:
            raise ValueError("|rho| must be <=1")
        self.rho = rho
        if theta < 0 or sigma < 0 or kappa < 0:
            raise ValueError("sigma,theta,kappa must be positive")
        else:
            self.theta = theta
            self.sigma = sigma
            self.kappa = kappa

    def path(self, S0, v0, N, T=1):
        """
        Produces one path of the Heston process.
        N = number of time steps
        T = Time in years
        Returns two arrays S (price) and v (variance).
        """

        MU = np.array([0, 0])
        COV = np.matrix([[1, self.rho], [self.rho, 1]])
        W = ss.multivariate_normal.rvs(mean=MU, cov=COV, size=N - 1)
        W_S = W[:, 0]  # Stock Brownian motion:     W_1
        W_v = W[:, 1]  # Variance Brownian motion:  W_2

        # Initialize vectors
        T_vec, dt = np.linspace(0, T, N, retstep=True)
        dt_sq = np.sqrt(dt)

        X0 = np.log(S0)
        v = np.zeros(N)
        v[0] = v0
        X = np.zeros(N)
        X[0] = X0

        # Generate paths
        for t in range(0, N - 1):
            v_sq = np.sqrt(v[t])
            v[t + 1] = np.abs(v[t] + self.kappa * (self.theta - v[t]) * dt + self.sigma * v_sq * dt_sq * W_v[t])
            X[t + 1] = X[t] + (self.mu - 0.5 * v[t]) * dt + v_sq * dt_sq * W_S[t]

        return np.exp(X), v


class NIG_process:
    """
    Class for the Normal Inverse Gaussian process:
    r = risk free constant rate
    Using the representation of Brownian subordination, the parameters are:
        theta = drift of the Brownian motion
        sigma = standard deviation of the Brownian motion
        kappa = variance of the of the Gamma process
    """

    def __init__(self, r=0.1, sigma=0.2, theta=-0.1, kappa=0.1):
        self.r = r
        self.theta = theta
        if sigma < 0 or kappa < 0:
            raise ValueError("sigma and kappa must be positive")
        else:
            self.sigma = sigma
            self.kappa = kappa

        # moments
        self.var = self.sigma**2 + self.theta**2 * self.kappa
        self.skew = (3 * self.theta**3 * self.kappa**2 + 3 * self.sigma**2 * self.theta * self.kappa) / (
            self.var ** (1.5)
        )
        self.kurt = (
            3 * self.sigma**4 * self.kappa
            + 18 * self.sigma**2 * self.theta**2 * self.kappa**2
            + 15 * self.theta**4 * self.kappa**3
        ) / (self.var**2)

    def exp_RV(self, S0, T, N):
        lam = T**2 / self.kappa  # scale for the IG process
        mu_s = T / lam  # scaled mean
        w = (1 - np.sqrt(1 - 2 * self.theta * self.kappa - self.kappa * self.sigma**2)) / self.kappa
        IG = ss.invgauss.rvs(mu=mu_s, scale=lam, size=N)  # The IG RV
        Norm = ss.norm.rvs(0, 1, N)  # The normal RV
        X = self.theta * IG + self.sigma * np.sqrt(IG) * Norm  # NIG random vector
        S_T = S0 * np.exp((self.r - w) * T + X)  # exponential dynamics
        return S_T.reshape((N, 1))


class GARCH:
    """
    Class for the GARCH(1,1) process. Variance process:

        V(t) = omega + alpha R^2(t-1) + beta V(t-1)

        VL:  Unconditional variance >=0
        alpha: coefficient > 0
        beta:  coefficient > 0
        gamma = 1 - alpha - beta
        omega = gamma*VL
    """

    def __init__(self, VL=0.04, alpha=0.08, beta=0.9):
        if VL < 0 or alpha <= 0 or beta <= 0:
            raise ValueError("VL>=0, alpha>0 and beta>0")
        else:
            self.VL = VL
            self.alpha = alpha
            self.beta = beta
        self.gamma = 1 - self.alpha - self.beta
        self.omega = self.gamma * self.VL

    def path(self, N=1000):
        """
        Generates a path with N points.
        Returns the return process R and the variance process var
        """
        eps = ss.norm.rvs(loc=0, scale=1, size=N)
        R = np.zeros_like(eps)
        var = np.zeros_like(eps)
        for i in range(N):
            var[i] = self.omega + self.alpha * R[i - 1] ** 2 + self.beta * var[i - 1]
            R[i] = np.sqrt(var[i]) * eps[i]
        return R, var

    def fit_from_data(self, data, disp=True):
        """
        MLE estimator for the GARCH
        """
        # Automatic re-scaling:
        # 1. the solver has problems with positive derivative in linesearch.
        # 2. the log has overflows using small values
        n = np.floor(np.log10(np.abs(data.mean())))
        R = data / 10**n

        # initial guesses
        a0 = 0.05
        b0 = 0.9
        g0 = 1 - a0 - b0
        w0 = g0 * np.var(R)

        # bounds and constraint
        bounds = ((0, None), (0, 1), (0, 1))

        def sum_small_1(x):
            return 1 - x[1] - x[2]

        cons = {"fun": sum_small_1, "type": "ineq"}

        def log_likely(x):
            var = R[0] ** 2  # initial variance
            N = len(R)
            log_lik = 0
            for i in range(1, N):
                var = x[0] + x[1] * R[i - 1] ** 2 + x[2] * var  # variance update
                log_lik += -np.log(var) - (R[i] ** 2 / var)
            return (-1) * log_lik

        result = minimize(
            log_likely,
            x0=[w0, a0, b0],
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            tol=1e-8,
            options={"maxiter": 150},
        )
        print(result.message)
        self.omega = result.x[0] * 10 ** (2 * n)
        self.alpha, self.beta = result.x[1:]
        self.gamma = 1 - self.alpha - self.beta
        self.VL = self.omega / self.gamma

        if disp is True:
            hess = approx_hess(result.x, log_likely)  # hessian by finite differences
            se = np.sqrt(np.diag(np.linalg.inv(hess)))  # standard error
            cv = ss.norm.ppf(1.0 - 0.05 / 2.0)  # alpha=0.05
            p_val = ss.norm.sf(np.abs(result.x / se))  # survival function

            df = pd.DataFrame(index=["omega", "alpha", "beta"])
            df["Params"] = result.x
            df["SE"] = se
            df["P-val"] = p_val
            df["95% CI lower"] = result.x - cv * se
            df["95% CI upper"] = result.x + cv * se
            df.loc["omega", ["Params", "SE", "95% CI lower", "95% CI upper"]] *= 10 ** (2 * n)
            print(df)

    def log_likelihood(self, R, last_var=True):
        """
        Computes the log-likelihood and optionally returns the last value
        of the variance
        """
        var = R[0] ** 2  # initial variance
        N = len(R)
        log_lik = 0
        log_2pi = np.log(2 * np.pi)
        for i in range(1, N):
            var = self.omega + self.alpha * R[i - 1] ** 2 + self.beta * var  # variance update
            log_lik += 0.5 * (-log_2pi - np.log(var) - (R[i] ** 2 / var))
        if last_var is True:
            return log_lik, var
        else:
            return log_lik

    def generate_var(self, R, R0, var0):
        """
        generate the variance process.
        R (array): return array
        R0: initial value of the returns
        var0: initial value of the variance
        """
        N = len(R)
        var = np.zeros(N)
        var[0] = self.omega + self.alpha * (R0**2) + self.beta * var0
        for i in range(1, N):
            var[i] = self.omega + self.alpha * R[i - 1] ** 2 + self.beta * var[i - 1]
        return var


class OU_process:
    """
    Class for the OU process:
    theta = long term mean
    sigma = diffusion coefficient
    kappa = mean reversion coefficient
    """

    def __init__(self, sigma=0.2, theta=-0.1, kappa=0.1):
        self.theta = theta
        if sigma < 0 or kappa < 0:
            raise ValueError("sigma,theta,kappa must be positive")
        else:
            self.sigma = sigma
            self.kappa = kappa

    def path(self, X0=0, T=1, N=10000, paths=1):
        """
        Produces a matrix of OU process:  X[N, paths]
        X0 = starting point
        N = number of time points (there are N-1 time steps)
        T = Time in years
        paths = number of paths
        """

        dt = T / (N - 1)
        X = np.zeros((N, paths))
        X[0, :] = X0
        W = ss.norm.rvs(loc=0, scale=1, size=(N - 1, paths))

        std_dt = np.sqrt(self.sigma**2 / (2 * self.kappa) * (1 - np.exp(-2 * self.kappa * dt)))
        for t in range(0, N - 1):
            X[t + 1, :] = self.theta + np.exp(-self.kappa * dt) * (X[t, :] - self.theta) + std_dt * W[t, :]

        return X


# In[21]:


def Thomas(A, b):
    """
    Solver for the linear equation Ax=b using the Thomas algorithm.
    It is a wrapper of the LAPACK function dgtsv.
    """

    D = A.diagonal(0)
    L = A.diagonal(-1)
    U = A.diagonal(1)

    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("expected square matrix")
    if A.shape[0] != b.shape[0]:
        raise ValueError("incompatible dimensions")

    (dgtsv,) = get_lapack_funcs(("gtsv",))
    du2, d, du, x, info = dgtsv(L, D, U, b)

    if info == 0:
        return x
    if info > 0:
        raise LinAlgError("singular matrix: resolution failed at diagonal %d" % (info - 1))


def SOR(A, b, w=1, eps=1e-10, N_max=100):
    """
    Solver for the linear equation Ax=b using the SOR algorithm.
          A = L + D + U
    Arguments:
        L = Strict Lower triangular matrix
        D = Diagonal
        U = Strict Upper triangular matrix
        w = Relaxation coefficient
        eps = tollerance
        N_max = Max number of iterations
    """

    x0 = b.copy()  # initial guess

    if sparse.issparse(A):
        D = sparse.diags(A.diagonal())  # diagonal
        U = sparse.triu(A, k=1)  # Strict U
        L = sparse.tril(A, k=-1)  # Strict L
        DD = (w * L + D).toarray()
    else:
        D = np.eye(A.shape[0]) * np.diag(A)  # diagonal
        U = np.triu(A, k=1)  # Strict U
        L = np.tril(A, k=-1)  # Strict L
        DD = w * L + D

    for i in range(1, N_max + 1):
        x_new = solve_triangular(DD, (w * b - w * U @ x0 - (w - 1) * D @ x0), lower=True)
        if norm(x_new - x0) < eps:
            return x_new
        x0 = x_new
        if i == N_max:
            raise ValueError("Fail to converge in {} iterations".format(i))


def SOR2(A, b, w=1, eps=1e-10, N_max=100):
    """
    Solver for the linear equation Ax=b using the SOR algorithm.
    It uses the coefficients and not the matrix multiplication.
    """
    N = len(b)
    x0 = np.ones_like(b, dtype=np.float64)  # initial guess
    x_new = np.ones_like(x0)  # new solution

    for k in range(1, N_max + 1):
        for i in range(N):
            S = 0
            for j in range(N):
                if j != i:
                    S += A[i, j] * x_new[j]
            x_new[i] = (1 - w) * x_new[i] + (w / A[i, i]) * (b[i] - S)

        if norm(x_new - x0) < eps:
            return x_new
        x0 = x_new.copy()
        if k == N_max:
            print("Fail to converge in {} iterations".format(k))


# In[22]:


class VG_pricer:
    """
    Closed Formula.
    Monte Carlo.
    Finite-difference PIDE: Explicit-implicit scheme, with Brownian approximation

        0 = dV/dt + (r -(1/2)sig^2 -w) dV/dx + (1/2)sig^2 d^V/dx^2
                 + \int[ V(x+y) nu(dy) ] -(r+lam)V
    """

    def __init__(self, Option_info, Process_info):
        """
        Process_info:  of type VG_process.
        It contains the interest rate r and the VG parameters (sigma, theta, kappa)

        Option_info:  of type Option_param.
        It contains (S0,K,T) i.e. current price, strike, maturity in years
        """
        self.r = Process_info.r  # interest rate
        self.sigma = Process_info.sigma  # VG parameter
        self.theta = Process_info.theta  # VG parameter
        self.kappa = Process_info.kappa  # VG parameter
        self.exp_RV = Process_info.exp_RV  # function to generate exponential VG Random Variables
        self.w = -np.log(1 - self.theta * self.kappa - self.kappa / 2 * self.sigma**2) / self.kappa  # coefficient w

        self.S0 = Option_info.S0  # current price
        self.K = Option_info.K  # strike
        self.T = Option_info.T  # maturity in years

        self.price = 0
        self.S_vec = None
        self.price_vec = None
        self.mesh = None
        self.exercise = Option_info.exercise
        self.payoff = Option_info.payoff

    def payoff_f(self, S):
        if self.payoff == "call":
            Payoff = np.maximum(S - self.K, 0)
        elif self.payoff == "put":
            Payoff = np.maximum(self.K - S, 0)
        return Payoff

    def closed_formula(self):
        """
        VG closed formula.  Put is obtained by put/call parity.
        """

        def Psy(a, b, g):
            f = lambda u: ss.norm.cdf(a / np.sqrt(u) + b * np.sqrt(u)) * u ** (g - 1) * np.exp(-u) / scps.gamma(g)
            result = quad(f, 0, np.inf)
            return result[0]

        # Ugly parameters
        xi = -self.theta / self.sigma**2
        s = self.sigma / np.sqrt(1 + ((self.theta / self.sigma) ** 2) * (self.kappa / 2))
        alpha = xi * s

        c1 = self.kappa / 2 * (alpha + s) ** 2
        c2 = self.kappa / 2 * alpha**2
        d = 1 / s * (np.log(self.S0 / self.K) + self.r * self.T + self.T / self.kappa * np.log((1 - c1) / (1 - c2)))

        # Closed formula
        call = self.S0 * Psy(
            d * np.sqrt((1 - c1) / self.kappa),
            (alpha + s) * np.sqrt(self.kappa / (1 - c1)),
            self.T / self.kappa,
        ) - self.K * np.exp(-self.r * self.T) * Psy(
            d * np.sqrt((1 - c2) / self.kappa),
            (alpha) * np.sqrt(self.kappa / (1 - c2)),
            self.T / self.kappa,
        )

        if self.payoff == "call":
            return call
        elif self.payoff == "put":
            return call - self.S0 + self.K * np.exp(-self.r * self.T)
        else:
            raise ValueError("invalid type. Set 'call' or 'put'")

    def Fourier_inversion(self):
        """
        Price obtained by inversion of the characteristic function
        """
        k = np.log(self.K / self.S0)  # log moneyness
        cf_VG_b = partial(
            cf_VG,
            t=self.T,
            mu=(self.r - self.w),
            theta=self.theta,
            sigma=self.sigma,
            kappa=self.kappa,
        )

        right_lim = 5000  # using np.inf may create warnings
        if self.payoff == "call":
            call = self.S0 * Q1(k, cf_VG_b, right_lim) - self.K * np.exp(-self.r * self.T) * Q2(
                k, cf_VG_b, right_lim
            )  # pricing function
            return call
        elif self.payoff == "put":
            put = self.K * np.exp(-self.r * self.T) * (1 - Q2(k, cf_VG_b, right_lim)) - self.S0 * (
                1 - Q1(k, cf_VG_b, right_lim)
            )  # pricing function
            return put
        else:
            raise ValueError("invalid type. Set 'call' or 'put'")

    def MC(self, N, Err=False, Time=False):
        """
        Variance Gamma Monte Carlo
        Err = return Standard Error if True
        Time = return execution time if True
        """
        t_init = time()

        S_T = self.exp_RV(self.S0, self.T, N)
        V = scp.mean(np.exp(-self.r * self.T) * self.payoff_f(S_T), axis=0)

        if Err is True:
            if Time is True:
                elapsed = time() - t_init
                return V, ss.sem(np.exp(-self.r * self.T) * self.payoff_f(S_T)), elapsed
            else:
                return V, ss.sem(np.exp(-self.r * self.T) * self.payoff_f(S_T))
        else:
            if Time is True:
                elapsed = time() - t_init
                return V, elapsed
            else:
                return V

    def FFT(self, K):
        """
        FFT method. It returns a vector of prices.
        K is an array of strikes
        """
        K = np.array(K)
        cf_VG_b = partial(
            cf_VG,
            t=self.T,
            mu=(self.r - self.w),
            theta=self.theta,
            sigma=self.sigma,
            kappa=self.kappa,
        )

        if self.payoff == "call":
            return fft_Lewis(K, self.S0, self.r, self.T, cf_VG_b, interp="cubic")
        elif self.payoff == "put":  # put-call parity
            return (
                fft_Lewis(K, self.S0, self.r, self.T, cf_VG_b, interp="cubic") - self.S0 + K * np.exp(-self.r * self.T)
            )
        else:
            raise ValueError("invalid type. Set 'call' or 'put'")

    def IV_Lewis(self):
        """Implied Volatility from the Lewis formula"""

        cf_VG_b = partial(
            cf_VG,
            t=self.T,
            mu=(self.r - self.w),
            theta=self.theta,
            sigma=self.sigma,
            kappa=self.kappa,
        )

        if self.payoff == "call":
            return IV_from_Lewis(self.K, self.S0, self.T, self.r, cf_VG_b)
        elif self.payoff == "put":
            raise NotImplementedError
        else:
            raise ValueError("invalid type. Set 'call' or 'put'")

    def PIDE_price(self, steps, Time=False):
        """
        steps = tuple with number of space steps and time steps
        payoff = "call" or "put"
        exercise = "European" or "American"
        Time = Boolean. Execution time.
        """
        t_init = time()

        Nspace = steps[0]
        Ntime = steps[1]

        S_max = 6 * float(self.K)
        S_min = float(self.K) / 6
        x_max = np.log(S_max)
        x_min = np.log(S_min)

        dev_X = np.sqrt(self.sigma**2 + self.theta**2 * self.kappa)  # std dev VG process

        dx = (x_max - x_min) / (Nspace - 1)
        extraP = int(np.floor(5 * dev_X / dx))  # extra points beyond the B.C.
        x = np.linspace(x_min - extraP * dx, x_max + extraP * dx, Nspace + 2 * extraP)  # space discretization
        t, dt = np.linspace(0, self.T, Ntime, retstep=True)  # time discretization

        Payoff = self.payoff_f(np.exp(x))
        offset = np.zeros(Nspace - 2)
        V = np.zeros((Nspace + 2 * extraP, Ntime))  # grid initialization

        if self.payoff == "call":
            V[:, -1] = Payoff  # terminal conditions
            V[-extraP - 1 :, :] = np.exp(x[-extraP - 1 :]).reshape(extraP + 1, 1) * np.ones(
                (extraP + 1, Ntime)
            ) - self.K * np.exp(-self.r * t[::-1]) * np.ones(
                (extraP + 1, Ntime)
            )  # boundary condition
            V[: extraP + 1, :] = 0
        else:
            V[:, -1] = Payoff
            V[-extraP - 1 :, :] = 0
            V[: extraP + 1, :] = self.K * np.exp(-self.r * t[::-1]) * np.ones((extraP + 1, Ntime))

        A = self.theta / (self.sigma**2)
        B = np.sqrt(self.theta**2 + 2 * self.sigma**2 / self.kappa) / self.sigma**2

        def levy_m(y):
            """Levy measure VG"""
            return np.exp(A * y - B * np.abs(y)) / (self.kappa * np.abs(y))

        eps = 1.5 * dx  # the cutoff near 0
        lam = (
            quad(levy_m, -(extraP + 1.5) * dx, -eps)[0] + quad(levy_m, eps, (extraP + 1.5) * dx)[0]
        )  # approximated intensity

        def int_w(y):
            """integrator"""
            return (np.exp(y) - 1) * levy_m(y)

        int_s = lambda y: np.abs(y) * np.exp(A * y - B * np.abs(y)) / self.kappa  # avoid division by zero

        w = (
            quad(int_w, -(extraP + 1.5) * dx, -eps)[0] + quad(int_w, eps, (extraP + 1.5) * dx)[0]
        )  # is the approx of omega

        sig2 = quad(int_s, -eps, eps)[0]  # the small jumps variance

        dxx = dx * dx
        a = (dt / 2) * ((self.r - w - 0.5 * sig2) / dx - sig2 / dxx)
        b = 1 + dt * (sig2 / dxx + self.r + lam)
        c = -(dt / 2) * ((self.r - w - 0.5 * sig2) / dx + sig2 / dxx)
        D = sparse.diags([a, b, c], [-1, 0, 1], shape=(Nspace - 2, Nspace - 2)).tocsc()
        DD = splu(D)

        nu = np.zeros(2 * extraP + 3)  # Lévy measure vector
        x_med = extraP + 1  # middle point in nu vector
        x_nu = np.linspace(-(extraP + 1 + 0.5) * dx, (extraP + 1 + 0.5) * dx, 2 * (extraP + 2))  # integration domain
        for i in range(len(nu)):
            if (i == x_med) or (i == x_med - 1) or (i == x_med + 1):
                continue
            nu[i] = quad(levy_m, x_nu[i], x_nu[i + 1])[0]

        if self.exercise == "European":
            # Backward iteration
            for i in range(Ntime - 2, -1, -1):
                offset[0] = a * V[extraP, i]
                offset[-1] = c * V[-1 - extraP, i]
                V_jump = V[extraP + 1 : -extraP - 1, i + 1] + dt * signal.convolve(
                    V[:, i + 1], nu[::-1], mode="valid", method="auto"
                )
                V[extraP + 1 : -extraP - 1, i] = DD.solve(V_jump - offset)
        elif self.exercise == "American":
            for i in range(Ntime - 2, -1, -1):
                offset[0] = a * V[extraP, i]
                offset[-1] = c * V[-1 - extraP, i]
                V_jump = V[extraP + 1 : -extraP - 1, i + 1] + dt * signal.convolve(
                    V[:, i + 1], nu[::-1], mode="valid", method="auto"
                )
                V[extraP + 1 : -extraP - 1, i] = np.maximum(DD.solve(V_jump - offset), Payoff[extraP + 1 : -extraP - 1])

        X0 = np.log(self.S0)  # current log-price
        self.S_vec = np.exp(x[extraP + 1 : -extraP - 1])  # vector of S
        self.price = np.interp(X0, x, V[:, 0])
        self.price_vec = V[extraP + 1 : -extraP - 1, 0]
        self.mesh = V[extraP + 1 : -extraP - 1, :]

        if Time is True:
            elapsed = time() - t_init
            return self.price, elapsed
        else:
            return self.price

    def plot(self, axis=None):
        if type(self.S_vec) != np.ndarray or type(self.price_vec) != np.ndarray:
            self.PIDE_price((5000, 4000))

        plt.plot(self.S_vec, self.payoff_f(self.S_vec), color="blue", label="Payoff")
        plt.plot(self.S_vec, self.price_vec, color="red", label="VG curve")
        if type(axis) == list:
            plt.axis(axis)
        plt.xlabel("S")
        plt.ylabel("price")
        plt.title("VG price")
        plt.legend(loc="upper left")
        plt.show()

    def mesh_plt(self):
        if type(self.S_vec) != np.ndarray or type(self.mesh) != np.ndarray:
            self.PDE_price((7000, 5000))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        X, Y = np.meshgrid(np.linspace(0, self.T, self.mesh.shape[1]), self.S_vec)
        ax.plot_surface(Y, X, self.mesh, cmap=cm.ocean)
        ax.set_title("VG price surface")
        ax.set_xlabel("S")
        ax.set_ylabel("t")
        ax.set_zlabel("V")
        ax.view_init(30, -100)  # this function rotates the 3d plot
        plt.show()

    def closed_formula_wrong(self):
        """
        VG closed formula. This implementation seems correct, BUT IT DOES NOT WORK!!
        Here I use the closed formula of Carr,Madan,Chang 1998.
        With scps.kv, a modified Bessel function of second kind.
        You can try to run it, but the output is slightly different from expected.
        """

        def Phi(alpha, beta, gamm, x, y):
            f = lambda u: u ** (alpha - 1) * (1 - u) ** (gamm - alpha - 1) * (1 - u * x) ** (-beta) * np.exp(u * y)
            result = quad(f, 0.00000001, 0.99999999)
            return (scps.gamma(gamm) / (scps.gamma(alpha) * scps.gamma(gamm - alpha))) * result[0]

        def Psy(a, b, g):
            c = np.abs(a) * np.sqrt(2 + b**2)
            u = b / np.sqrt(2 + b**2)

            value = (
                (c ** (g + 0.5) * np.exp(np.sign(a) * c) * (1 + u) ** g)
                / (np.sqrt(2 * np.pi) * g * scps.gamma(g))
                * scps.kv(g + 0.5, c)
                * Phi(g, 1 - g, 1 + g, (1 + u) / 2, -np.sign(a) * c * (1 + u))
                - np.sign(a)
                * (c ** (g + 0.5) * np.exp(np.sign(a) * c) * (1 + u) ** (1 + g))
                / (np.sqrt(2 * np.pi) * (g + 1) * scps.gamma(g))
                * scps.kv(g - 0.5, c)
                * Phi(g + 1, 1 - g, 2 + g, (1 + u) / 2, -np.sign(a) * c * (1 + u))
                + np.sign(a)
                * (c ** (g + 0.5) * np.exp(np.sign(a) * c) * (1 + u) ** (1 + g))
                / (np.sqrt(2 * np.pi) * (g + 1) * scps.gamma(g))
                * scps.kv(g - 0.5, c)
                * Phi(g, 1 - g, 1 + g, (1 + u) / 2, -np.sign(a) * c * (1 + u))
            )
            return value

        # Ugly parameters
        xi = -self.theta / self.sigma**2
        s = self.sigma / np.sqrt(1 + ((self.theta / self.sigma) ** 2) * (self.kappa / 2))
        alpha = xi * s

        c1 = self.kappa / 2 * (alpha + s) ** 2
        c2 = self.kappa / 2 * alpha**2
        d = 1 / s * (np.log(self.S0 / self.K) + self.r * self.T + self.T / self.kappa * np.log((1 - c1) / (1 - c2)))

        # Closed formula
        call = self.S0 * Psy(
            d * np.sqrt((1 - c1) / self.kappa),
            (alpha + s) * np.sqrt(self.kappa / (1 - c1)),
            self.T / self.kappa,
        ) - self.K * np.exp(-self.r * self.T) * Psy(
            d * np.sqrt((1 - c2) / self.kappa),
            (alpha) * np.sqrt(self.kappa / (1 - c2)),
            self.T / self.kappa,
        )

        return call


# In[23]:


class BS_pricer:
    """
    Closed Formula.
    Monte Carlo.
    Finite-difference Black-Scholes PDE:
     df/dt + r df/dx + 1/2 sigma^2 d^f/dx^2 -rf = 0
    """

    def __init__(self, Option_info, Process_info):
        """
        Option_info: of type Option_param. It contains (S0,K,T)
                i.e. current price, strike, maturity in years
        Process_info: of type Diffusion_process. It contains (r, mu, sig) i.e.
                interest rate, drift coefficient, diffusion coefficient
        """
        self.r = Process_info.r  # interest rate
        self.sig = Process_info.sig  # diffusion coefficient
        self.S0 = Option_info.S0  # current price
        self.K = Option_info.K  # strike
        self.T = Option_info.T  # maturity in years
        self.exp_RV = Process_info.exp_RV  # function to generate solution of GBM

        self.price = 0
        self.S_vec = None
        self.price_vec = None
        self.mesh = None
        self.exercise = Option_info.exercise
        self.payoff = Option_info.payoff

    def payoff_f(self, S):
        if self.payoff == "call":
            Payoff = np.maximum(S - self.K, 0)
        elif self.payoff == "put":
            Payoff = np.maximum(self.K - S, 0)
        return Payoff

    @staticmethod
    def BlackScholes(payoff="call", S0=100.0, K=100.0, T=1.0, r=0.1, sigma=0.2):
        """Black Scholes closed formula:
        payoff: call or put.
        S0: float.    initial stock/index level.
        K: float strike price.
        T: float maturity (in year fractions).
        r: float constant risk-free short rate.
        sigma: volatility factor in diffusion term."""

        d1 = (np.log(S0 / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = (np.log(S0 / K) + (r - sigma**2 / 2) * T) / (sigma * np.sqrt(T))

        if payoff == "call":
            return S0 * ss.norm.cdf(d1) - K * np.exp(-r * T) * ss.norm.cdf(d2)
        elif payoff == "put":
            return K * np.exp(-r * T) * ss.norm.cdf(-d2) - S0 * ss.norm.cdf(-d1)
        else:
            raise ValueError("invalid type. Set 'call' or 'put'")

    @staticmethod
    def vega(sigma, S0, K, T, r):
        """BS vega: derivative of the price with respect to the volatility"""
        d1 = (np.log(S0 / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
        return S0 * np.sqrt(T) * ss.norm.pdf(d1)

    def closed_formula(self):
        """
        Black Scholes closed formula:
        """
        d1 = (np.log(self.S0 / self.K) + (self.r + self.sig**2 / 2) * self.T) / (self.sig * np.sqrt(self.T))
        d2 = (np.log(self.S0 / self.K) + (self.r - self.sig**2 / 2) * self.T) / (self.sig * np.sqrt(self.T))

        if self.payoff == "call":
            return self.S0 * ss.norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * ss.norm.cdf(d2)
        elif self.payoff == "put":
            return self.K * np.exp(-self.r * self.T) * ss.norm.cdf(-d2) - self.S0 * ss.norm.cdf(-d1)
        else:
            raise ValueError("invalid type. Set 'call' or 'put'")
    def Fourier_inversion(self):
        """
        Price obtained by inversion of the characteristic function
        """
        k = np.log(self.K / self.S0)
        cf_GBM = partial(
            cf_normal,
            mu=(self.r - 0.5 * self.sig**2) * self.T,
            sig=self.sig * np.sqrt(self.T),
        )  # function binding

        if self.payoff == "call":
            call = self.S0 * calc_Q1(k, cf_GBM, np.inf) - self.K * np.exp(-self.r * self.T) * calc_Q2(
                k, cf_GBM, np.inf
            )  # pricing function
            return call
        elif self.payoff == "put":
            put = self.K * np.exp(-self.r * self.T) * (1 - calc_Q2(k, cf_GBM, np.inf)) - self.S0 * (
                1 - calc_Q1(k, cf_GBM, np.inf)
            )  # pricing function
            return put
        else:
            raise ValueError("invalid type. Set 'call' or 'put'")  
    
    def MC(self, N, Err=False, Time=False):
        """
        BS Monte Carlo
        Err = return Standard Error if True
        Time = return execution time if True
        """
        t_init = time()

        S_T = self.exp_RV(self.S0, self.T, N)
        PayOff = self.payoff_f(S_T)
        V = scp.mean(np.exp(-self.r * self.T) * PayOff, axis=0)

        if Err is True:
            if Time is True:
                elapsed = time() - t_init
                return V, ss.sem(np.exp(-self.r * self.T) * self.payoff_f(S_T)), elapsed
            else:
                return V, ss.sem(np.exp(-self.r * self.T) * self.payoff_f(S_T))
        else:
            if Time is True:
                elapsed = time() - t_init
                return V, elapsed
            else:
                return V
        
    def PDE_price(self, steps, Time=False, solver="splu"):
        """
        steps = tuple with number of space steps and time steps
        payoff = "call" or "put"
        exercise = "European" or "American"
        Time = Boolean. Execution time.
        Solver = spsolve or splu or Thomas or SOR
        """
        t_init = time()

        Nspace = steps[0]
        Ntime = steps[1]

        S_max = 6 * float(self.K)
        S_min = float(self.K) / 6
        x_max = np.log(S_max)
        x_min = np.log(S_min)
        x0 = np.log(self.S0)  # current log-price

        x, dx = np.linspace(x_min, x_max, Nspace, retstep=True)
        t, dt = np.linspace(0, self.T, Ntime, retstep=True)

        self.S_vec = np.exp(x)  # vector of S
        Payoff = self.payoff_f(self.S_vec)

        V = np.zeros((Nspace, Ntime))
        if self.payoff == "call":
            V[:, -1] = Payoff
            V[-1, :] = np.exp(x_max) - self.K * np.exp(-self.r * t[::-1])
            V[0, :] = 0
        else:
            V[:, -1] = Payoff
            V[-1, :] = 0
            V[0, :] = Payoff[0] * np.exp(-self.r * t[::-1])  # Instead of Payoff[0] I could use K
            # For s to 0, the limiting value is e^(-rT)(K-s)

        sig2 = self.sig**2
        dxx = dx**2
        a = (dt / 2) * ((self.r - 0.5 * sig2) / dx - sig2 / dxx)
        b = 1 + dt * (sig2 / dxx + self.r)
        c = -(dt / 2) * ((self.r - 0.5 * sig2) / dx + sig2 / dxx)

        D = sparse.diags([a, b, c], [-1, 0, 1], shape=(Nspace - 2, Nspace - 2)).tocsc()

        offset = np.zeros(Nspace - 2)

        if solver == "spsolve":
            if self.exercise == "European":
                for i in range(Ntime - 2, -1, -1):
                    offset[0] = a * V[0, i]
                    offset[-1] = c * V[-1, i]
                    V[1:-1, i] = spsolve(D, (V[1:-1, i + 1] - offset))
            elif self.exercise == "American":
                for i in range(Ntime - 2, -1, -1):
                    offset[0] = a * V[0, i]
                    offset[-1] = c * V[-1, i]
                    V[1:-1, i] = np.maximum(spsolve(D, (V[1:-1, i + 1] - offset)), Payoff[1:-1])
        elif solver == "Thomas":
            if self.exercise == "European":
                for i in range(Ntime - 2, -1, -1):
                    offset[0] = a * V[0, i]
                    offset[-1] = c * V[-1, i]
                    V[1:-1, i] = Thomas(D, (V[1:-1, i + 1] - offset))
            elif self.exercise == "American":
                for i in range(Ntime - 2, -1, -1):
                    offset[0] = a * V[0, i]
                    offset[-1] = c * V[-1, i]
                    V[1:-1, i] = np.maximum(Thomas(D, (V[1:-1, i + 1] - offset)), Payoff[1:-1])
        elif solver == "SOR":
            if self.exercise == "European":
                for i in range(Ntime - 2, -1, -1):
                    offset[0] = a * V[0, i]
                    offset[-1] = c * V[-1, i]
                    V[1:-1, i] = SOR(a, b, c, (V[1:-1, i + 1] - offset), w=1.68, eps=1e-10, N_max=600)
            elif self.exercise == "American":
                for i in range(Ntime - 2, -1, -1):
                    offset[0] = a * V[0, i]
                    offset[-1] = c * V[-1, i]
                    V[1:-1, i] = np.maximum(
                        SOR(
                            a,
                            b,
                            c,
                            (V[1:-1, i + 1] - offset),
                            w=1.68,
                            eps=1e-10,
                            N_max=600,
                        ),
                        Payoff[1:-1],
                    )
        elif solver == "splu":
            DD = splu(D)
            if self.exercise == "European":
                for i in range(Ntime - 2, -1, -1):
                    offset[0] = a * V[0, i]
                    offset[-1] = c * V[-1, i]
                    V[1:-1, i] = DD.solve(V[1:-1, i + 1] - offset)
            elif self.exercise == "American":
                for i in range(Ntime - 2, -1, -1):
                    offset[0] = a * V[0, i]
                    offset[-1] = c * V[-1, i]
                    V[1:-1, i] = np.maximum(DD.solve(V[1:-1, i + 1] - offset), Payoff[1:-1])
        else:
            raise ValueError("Solver is splu, spsolve, SOR or Thomas")

        self.price = np.interp(x0, x, V[:, 0])
        self.price_vec = V[:, 0]
        self.mesh = V

        if Time is True:
            elapsed = time() - t_init
            return self.price, elapsed
        else:
            return self.price

    def plot(self, axis=None):
        if type(self.S_vec) != np.ndarray or type(self.price_vec) != np.ndarray:
            self.PDE_price((7000, 5000))
            # print("run the PDE_price method")
            # return

        plt.plot(self.S_vec, self.payoff_f(self.S_vec), color="blue", label="Payoff")
        plt.plot(self.S_vec, self.price_vec, color="red", label="BS curve")
        if type(axis) == list:
            plt.axis(axis)
        plt.xlabel("S")
        plt.ylabel("price")
        plt.title(f"{self.exercise} - Black Scholes price")
        plt.legend()
        plt.show()

    def mesh_plt(self):
        if type(self.S_vec) != np.ndarray or type(self.mesh) != np.ndarray:
            self.PDE_price((7000, 5000))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        X, Y = np.meshgrid(np.linspace(0, self.T, self.mesh.shape[1]), self.S_vec)
        ax.plot_surface(Y, X, self.mesh, cmap=cm.ocean)
        ax.set_title(f"{self.exercise} - BS price surface")
        ax.set_xlabel("S")
        ax.set_ylabel("t")
        ax.set_zlabel("V")
        ax.view_init(30, -100)  # this function rotates the 3d plot
        plt.show()

    def LSM(self, N=10000, paths=10000, order=2):
        """
        Longstaff-Schwartz Method for pricing American options

        N = number of time steps
        paths = number of generated paths
        order = order of the polynomial for the regression
        """

        if self.payoff != "put":
            raise ValueError("invalid type. Set 'call' or 'put'")

        dt = self.T / (N - 1)  # time interval
        df = np.exp(-self.r * dt)  # discount factor per time time interval

        X0 = np.zeros((paths, 1))
        increments = ss.norm.rvs(
            loc=(self.r - self.sig**2 / 2) * dt,
            scale=np.sqrt(dt) * self.sig,
            size=(paths, N - 1),
        )
        X = np.concatenate((X0, increments), axis=1).cumsum(1)
        S = self.S0 * np.exp(X)

        H = np.maximum(self.K - S, 0)  # intrinsic values for put option
        V = np.zeros_like(H)  # value matrix
        V[:, -1] = H[:, -1]

        # Valuation by LS Method
        for t in range(N - 2, 0, -1):
            good_paths = H[:, t] > 0
            rg = np.polyfit(S[good_paths, t], V[good_paths, t + 1] * df, 2)  # polynomial regression
            C = np.polyval(rg, S[good_paths, t])  # evaluation of regression

            exercise = np.zeros(len(good_paths), dtype=bool)
            exercise[good_paths] = H[good_paths, t] > C

            V[exercise, t] = H[exercise, t]
            V[exercise, t + 1 :] = 0
            discount_path = V[:, t] == 0
            V[discount_path, t] = V[discount_path, t + 1] * df

        V0 = np.mean(V[:, 1]) * df  #
        return V0
    
    def FFT(self, K):
        """
        FFT method. It returns a vector of prices.
        K is an array of strikes
        """
        K = np.array(K)
        cf_GBM = partial(
            cf_normal,
            mu=(self.r - 0.5 * self.sig**2) * self.T,
            sig=self.sig * np.sqrt(self.T),
        )  # function binding
        if self.payoff == "call":
            return fft_Lewis(K, self.S0, self.r, self.T, cf_GBM, interp="cubic")
        elif self.payoff == "put":  # put-call parity
            return (
                fft_Lewis(K, self.S0, self.r, self.T, cf_GBM, interp="cubic") - self.S0 + K * np.exp(-self.r * self.T)
            )
        else:
            raise ValueError("invalid type. Set 'call' or 'put'")

    def IV_Lewis(self):
        """Implied Volatility from the Lewis formula"""

        cf_GBM = partial(
            cf_normal,
            mu=(self.r - 0.5 * self.sig**2) * self.T,
            sig=self.sig * np.sqrt(self.T),
        )  # function binding
        if self.payoff == "call":
            return IV_from_Lewis(self.K, self.S0, self.T, self.r, cf_GBM)
        elif self.payoff == "put":
            raise NotImplementedError
        else:
            raise ValueError("invalid type. Set 'call' or 'put'")


# In[ ]:




