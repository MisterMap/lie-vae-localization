import numpy as np
import torch
import math


def reparametrize(z_mu, z_logvar):
    epsilon = torch.randn_like(z_mu)
    z = z_mu + torch.exp(0.5 * z_logvar) * epsilon
    return z


def kl(z_mean, z_logvar):
    kl_divergence_element = -0.5 * (-z_mean ** 2 - torch.exp(z_logvar) + 1 + z_logvar)
    return kl_divergence_element.sum()


def calculate_distribution_product(mu, logvar):
    log_denominator = torch.logsumexp(-logvar, dim=0)
    result_logvar = -log_denominator
    weights = torch.exp(-logvar - log_denominator[None])
    result_mu = torch.sum(mu * weights, dim=0)
    return result_mu, result_logvar


def add_prefix(prefix, dictionary):
    result = {}
    for key, value in dictionary.items():
        result[f"{prefix}_{key}"] = value
    return result


def deregularize_normal_distribution(z_mu, z_logvar):
    mu = torch.cat([z_mu[None], torch.zeros_like(z_mu)[None]], dim=0)
    logvar = torch.cat([z_logvar[None], torch.zeros_like(z_logvar)[None]], dim=0)
    z_mu, z_logvar = calculate_distribution_product(mu, logvar)
    return z_mu, z_logvar

