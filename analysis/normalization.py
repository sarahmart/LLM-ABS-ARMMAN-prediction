import numpy as np
from scipy.stats import rankdata


def z_score_normalization(sigma2):
    """
    Normalize uncertainties using Z-score normalization.
    
    Args:
    - sigma2: Array of uncertainty values for the model.
    
    Returns:
    - sigma2_norm: Z-score normalized uncertainties.
    """
    mean = np.mean(sigma2)
    std_dev = np.std(sigma2)
    
    # Z-score normalization
    sigma2_norm = (sigma2 - mean) / std_dev
    return sigma2_norm


def log_normalization(sigma2):
    """
    Normalize uncertainties using logarithmic normalization.
    
    Args:
    - sigma2: Array of uncertainty values for the model.
    
    Returns:
    - sigma2_norm: Logarithmic normalized uncertainties.
    """
    # Apply log transformation
    sigma2_norm = np.log(1 + sigma2)
    return sigma2_norm


def rank_normalization(sigma2):
    """
    Normalize uncertainties using percentile rank normalization.
    (focus on the relative ranking/order of values)
    
    Args:
    - sigma2: Array of uncertainty values for the model.
    
    Returns:
    - sigma2_norm: Percentile rank normalized uncertainties.
    """
    # Get rank of each value in sigma2
    ranks = rankdata(sigma2, method='min')
    # Normalize ranks to a [0, 1] range
    sigma2_norm = ranks / len(sigma2)
    return sigma2_norm


def min_max_normalization(sigma2_LLM, sigma2_MC):
    """
    Normalize uncertainties for both models together using min-max normalization.
    
    Args:
    - sigma2_LLM: Array of LLM uncertainties.
    - sigma2_MC: Array of MC uncertainties.
    
    Returns:
    - sigma2_LLM_norm: Normalized LLM uncertainties.
    - sigma2_MC_norm: Normalized MC uncertainties.
    """
    combined_min = min(np.min(sigma2_LLM), np.min(sigma2_MC))
    combined_max = max(np.max(sigma2_LLM), np.max(sigma2_MC))

    sigma2_LLM_norm = (sigma2_LLM - combined_min) / (combined_max - combined_min)
    sigma2_MC_norm = (sigma2_MC - combined_min) / (combined_max - combined_min)
    
    return sigma2_LLM_norm, sigma2_MC_norm
