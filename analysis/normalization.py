import numpy as np
from scipy.stats import rankdata


def z_score_normalization(uncertainties):
    """
    Normalize uncertainties across models using Z-score normalization.

    Args:
    - uncertainties: List of arrays, where each array represents the uncertainty values for a model 
                     (each array should have shape [num_samples]).

    Returns:
    - normalized_uncertainties: List of arrays, where each array represents Z-score normalized uncertainties
                                across models for each mother and timestep.
    """
    
    # Convert list of arrays into 2D array (models x samples)
    uncertainties_array = np.array(uncertainties)

    # Compute mean and std deviation across models
    mean = np.mean(uncertainties_array, axis=0)
    std_dev = np.std(uncertainties_array, axis=0)

    # Z-score normalization
    z_scores = (uncertainties_array - mean) / (std_dev + 1e-10)

    return [z_scores[i, :] for i in range(z_scores.shape[0])]


def log_normalization(uncertainties):
    """
    Normalize uncertainties across models using log normalization.

    Args:
    - uncertainties: List of arrays, where each array represents the uncertainty values for a model 
                     (each array should have shape [num_samples]).

    Returns:
    - normalized_uncertainties: List of arrays, where each array represents logarithmically normalized uncertainties
                                across models for each mother and timestep.
    """

    # Convert list of arrays to 2D array (models x samples)
    uncertainties_array = np.array(uncertainties) 

    # Apply log transformation
    log_normalized = np.log(1 + uncertainties_array)

    return [log_normalized[i, :] for i in range(log_normalized.shape[0])]


def rank_normalization(uncertainties):
    """
    Normalize uncertainties across models for each mother and timestep using percentile rank normalization.

    Args:
    - uncertainties: List of arrays, where each array represents the uncertainty values for a model 
                     (each array should have shape [num_samples]).

    Returns:
    - normalized_uncertainties: List of arrays, where each array represents the normalized uncertainties for a model,
                                normalized across models for each mother and timestep.
    """
    # Convert list of arrays to 2D array w shape (num_models, num_samples)
    uncertainties_array = np.array(uncertainties)  
    
    # rank normalization across models for each sample
    ranks = np.apply_along_axis(rankdata, axis=0, arr=uncertainties_array, method='min')
    
    # Normalize ranks to [0, 1] for each sample
    normalized_uncertainties = ranks / ranks.shape[0]
    
    return [normalized_uncertainties[i, :] for i in range(normalized_uncertainties.shape[0])] 


def min_max_normalization(uncertainties):
    """
    Normalize uncertainties across all models using min-max normalization.

    Args:
    - uncertainties: List of arrays, where each array represents the uncertainty values for a model 
                     (each array should have shape [num_samples]).

    Returns:
    - normalized_uncertainties: List of arrays, where each array represents normalized uncertainties
                                for a model (normalized across all models).
    """

    # Convert list of arrays into a 2D array (models x samples)
    uncertainties_array = np.array(uncertainties)  
    
    # Calculate global min and max
    combined_min = np.min(uncertainties_array)
    combined_max = np.max(uncertainties_array)

    normalized_uncertainties_array = (uncertainties_array - combined_min) / (combined_max - combined_min)

    return [normalized_uncertainties_array[i, :] for i in range(normalized_uncertainties_array.shape[0])]


def min_max_normalization_per_timestep(uncertainties):
    """
    Normalize uncertainties across models for each timestep using min-max normalization.

    Args:
    - uncertainties: List of arrays, where each array represents the uncertainty values for a model 
                     (each array should have shape [num_samples]).

    Returns:
    - normalized_uncertainties: List of arrays, where each array represents normalized uncertainties
                                for a model, normalized across models for each timestep.
    """
    # Convert list of arrays into a 2D array (models x samples)
    uncertainties_array = np.array(uncertainties)  # Shape: (num_models, num_samples)

    # Compute min and max across models for each sample (axis=0)
    combined_min = np.min(uncertainties_array, axis=0)  # Min for each timestep
    combined_max = np.max(uncertainties_array, axis=0)  # Max for each timestep

    # Apply min-max normalization for each sample
    normalized_uncertainties_array = (uncertainties_array - combined_min) / (combined_max - combined_min + 1e-10)

    # Return as a list of arrays
    return [normalized_uncertainties_array[i, :] for i in range(normalized_uncertainties_array.shape[0])]
