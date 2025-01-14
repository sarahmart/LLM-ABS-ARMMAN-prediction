# imports
import numpy as np
import json
from sklearn.metrics import accuracy_score, f1_score


def load_predictions_and_ground_truths(predictions_path, ground_truths_path):
    """Load saved individual predictions and ground truths from JSON files."""
    with open(predictions_path, 'r') as f:
        all_individual_predictions = json.load(f)
    
    with open(ground_truths_path, 'r') as f:
        ground_truths = json.load(f)
    
    return all_individual_predictions, ground_truths


def compute_metrics(P_combined, ground_truths, threshold=0.5):
    """
    Compute accuracy, F1 score, and log likelihood for the aggregated predictions.
    
    Args:
    - P_combined: Aggregated posterior probability of engagement (array of shape [num_samples]).
    - ground_truths: Ground truth labels (binary, array of shape [num_samples]).
    - threshold: Probability threshold to classify predictions as engaged or not engaged.
    
    Returns:
    - accuracy: Accuracy of the predictions.
    - f1: F1 score of the predictions.
    - log_likelihood: Log likelihood of the predictions.
    """
    # Ensure ground truths are binary integers
    ground_truths = np.array(ground_truths).astype(int)
    
    # Binarize the predictions based on threshold and ensure they are integers
    P_combined = np.array(P_combined)
    predictions = (P_combined >= threshold).astype(int)

    # Check if predictions and ground_truths are actually binary (0 or 1)
    # print(f"Unique values in predictions: {np.unique(predictions)}")
    # print(f"Unique values in ground_truths: {np.unique(ground_truths)}")
    
    # Compute accuracy
    accuracy = accuracy_score(ground_truths, predictions)
    
    # Compute F1 score
    f1 = f1_score(ground_truths, predictions)
    
    # Compute log likelihood (binary cross-entropy)
    epsilon = 1e-10  # To avoid log(0) errors
    P_combined_clipped = np.clip(P_combined, epsilon, 1 - epsilon)
    log_likelihood = np.sum(
        ground_truths * np.log(P_combined_clipped) + (1 - ground_truths) * np.log(1 - P_combined_clipped)
    ) / len(ground_truths) # normalise by pop size
    
    return accuracy, f1, log_likelihood

def restructure_predictions(all_individual_predictions): # 2 * 2 * 5 * 5
    # matrix 1 : 1 * 2 * 5 * 5 
    # matrix 2 : 1 * 2 * 5 * 5
    """
    Restructure the list of individual predictions into a matrix for each timestep.
    
    Args:
    - all_individual_predictions: List of lists, where each sublist contains multiple predictions for a single arm at each timestep.
    
    Returns:
    - restructured_predictions: List of arrays, where each array contains the predictions for a single timestep.
    """
    # Extract matrices for each timestep
    num_timesteps = all_individual_predictions.shape[0]
    restructured_predictions = [all_individual_predictions[t, :, :, :] for t in range(num_timesteps)]

    return restructured_predictions


def binary_entropy(p):
    """Compute binary entropy for Bernoulli distribution."""
    p = np.clip(p, 1e-10, 1 - 1e-10)  # Avoid log(0) errors
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def compute_uncertainties_from_llm_predictions(all_individual_predictions):
    """
    Compute epistemic and aleatoric uncertainties from LLM predictions.
    
    Args:
    - all_individual_predictions: List of lists, where each sublist contains multiple predictions for a single arm.
    
    Returns:
    - epistemic_uncertainty: List of epistemic uncertainties for each arm.
    - aleatoric_uncertainty: List of aleatoric uncertainties for each arm.
    - predictive_uncertainty: List of predictive uncertainties for each arm.
    """
    epistemic_uncertainty = []
    aleatoric_uncertainty = []
    predictive_uncertainty = []

    for arm_predictions in all_individual_predictions:
        # Convert list of predictions into a numpy array (shape: [num_queries])
        arm_predictions = np.array(arm_predictions)

        # Mean prediction across queries for this arm
        mean_prediction = np.mean(arm_predictions)

        # Compute predictive uncertainty (entropy of the mean prediction)
        pred_uncertainty = binary_entropy(mean_prediction)
        predictive_uncertainty.append(pred_uncertainty)

        # Compute aleatoric uncertainty (mean entropy of individual predictions)
        individual_entropies = binary_entropy(arm_predictions)
        aleatoric_uncertainty.append(np.mean(individual_entropies))

        # Epistemic uncertainty is the difference between predictive and aleatoric uncertainty
        epistemic_uncertainty.append(pred_uncertainty - np.mean(individual_entropies))

    return epistemic_uncertainty, aleatoric_uncertainty, predictive_uncertainty


def infer_posterior(*predictions, uncertainties=None, normalization_method=None):
    """
    Infer the posterior probability using multiple models as priors (direct Bayesian averaging of results).

    Args:
    - *predictions: A variable number of arrays, where each array represents the probability of engagement
                    predicted by a model (each array should have shape [num_samples]).
    - uncertainties: Array for each model representing epistemic uncertainties (shape: [num_samples])

    Returns:
    - P_posterior: Posterior probability of engagement (array of shape [num_samples]).
    - avg_uncertainty: Average uncertainty across models w.out weighting (array of shape [num_samples]).
    """

    # Normalize uncertainties (if specified)
    if normalization_method is not None:
        uncertainties = [normalization_method(u) for u in uncertainties]

    # Start with first model's predictions as initial prior
    P_posterior = predictions[0]
    
    # Update posterior iteratively --> loop through all models' results
    for P in predictions[1:]:
        # Calculate probability of not engaging for current posterior and current model
        P_not_engage_posterior = 1 - P_posterior
        P_not_engage_current = 1 - P

        # Apply Bayesian update
        numerator = P_posterior * P
        denominator = (P_posterior * P) + (P_not_engage_posterior * P_not_engage_current)
        
        # Avoid zero division
        epsilon = 1e-10
        P_posterior = numerator / (denominator + epsilon)

    # Calculate average uncertainty across models (new lines added)
    if uncertainties is not None:
        avg_uncertainty = np.mean(uncertainties, axis=0)
    else:
        avg_uncertainty = None

    return P_posterior, avg_uncertainty


def uncertainty_based_selection(predictions, uncertainties):
    """
    Aggregate predictions by selecting the model with the lowest epistemic uncertainty.
    That lowest uncertainty value is the corresponding uncertainty for the now-aggregated prediction.

    Args:
    - predictions: List of arrays, where each array is the probability of engagement predicted by a model 
                   (shape [num_samples] for each array).
    - uncertainties: List of arrays, where each array is the epistemic uncertainty of the corresponding model's predictions 
                     (shape [num_samples] for each array).

    Returns:
    - P_combined: Combined posterior probability of engagement, selecting the lowest uncertainty for each sample 
                  (array of shape [num_samples]).
    # - P_combined_uncertainty: Lowest uncertainty value corresponding to the selected model prediction for each sample
    #                           (array of shape [num_samples]).
    """
    # Stack predictions and uncertainties into arrays of shape [num_samples, num_models]
    predictions = np.stack(predictions, axis=1)
    uncertainties = np.stack(uncertainties, axis=1)

    # Index of model with lowest uncertainty for each
    min_uncertainty_indices = np.argmin(uncertainties, axis=1)

    # Select lowest uncertainty predictions
    P_combined = predictions[np.arange(predictions.shape[0]), min_uncertainty_indices]
    
    return P_combined


def bayesian_aggregation(predictions, uncertainties, normalization_method=None):
    """
    Aggregate predictions from multiple models using Bayesian weighting with a linear combination.

    Args:
    - predictions: List of arrays, where each array is the probability of engagement predicted by a model 
                   (each array should have shape [num_samples]).
    - uncertainties: List of arrays, where each array is the epistemic uncertainty of the corresponding model's predictions 
                     (each array should have shape [num_samples]).
    - normalization_method: Function to normalize uncertainties from normalization.py.

    Returns:
    - P_combined: Combined posterior probability of engagement (array of shape [num_samples]).
    """

    # Normalize uncertainties (if specified)
    if normalization_method is not None:
        uncertainties = [normalization_method(u) for u in uncertainties]

    # Add small epsilon to avoid zero division w near-zero uncertainties
    epsilon = 1e-10
    uncertainties = [np.array(u) + epsilon for u in uncertainties]
    
    # Convert epistemic uncertainties to precision (tau)
    precisions = [1 / u for u in uncertainties]
    
    # Sum precisions to compute denominator for each point
    total_precision = np.sum(precisions, axis=0)

    # Calculate weighted predictions :
    # multiply each prediction with its model's precision, then sum 
    weighted_predictions = np.sum([p * tau for p, tau in zip(predictions, precisions)], axis=0)
    
    # Compute final combined prediction
    P_combined = weighted_predictions / total_precision
    P_combined = np.clip(P_combined, 0, 1)

    # Handle NaNs by replaceing w 0.5 prob
    if np.any(np.isnan(P_combined)):
        print("Warning: NaN values detected in P_combined. Replacing with 0.5.")
        P_combined = np.nan_to_num(P_combined, nan=0.5)  

    # Compute aggregated uncertainty (new lines added)
    weighted_uncertainties = np.sum([u * tau for u, tau in zip(uncertainties, precisions)], axis=0)
    unc_combined = weighted_uncertainties / total_precision
    
    return P_combined, unc_combined


def identify_discrepancies(P_LLM, P_MC, ground_truths, threshold=0.5):
    """
    Identify indices (arms) where one model (LLM or MC) is correct and the other is incorrect.
    
    Args:
    - P_LLM: Predictions from the LLM model.
    - P_MC: Predictions from the MC model.
    - ground_truths: Ground truth labels.
    - threshold: Threshold to classify predictions as engaged or not engaged.

    Returns:
    - correct_llm_incorrect_mc: Indices where LLM is correct and MC is incorrect.
    - correct_mc_incorrect_llm: Indices where MC is correct and LLM is incorrect.
    """
    predictions_llm = (P_LLM >= threshold).astype(int)
    predictions_mc = (P_MC >= threshold).astype(int)
    
    correct_llm = (predictions_llm == ground_truths)
    correct_mc = (predictions_mc == ground_truths)
    
    correct_llm_incorrect_mc = np.where((correct_llm == 1) & (correct_mc == 0))[0]
    correct_mc_incorrect_llm = np.where((correct_mc == 1) & (correct_llm == 0))[0]
    
    return correct_llm_incorrect_mc, correct_mc_incorrect_llm


def compare_confidence(correct_indices, P_LLM, P_MC, sigma2_LLM, sigma2_MC, ground_truths, eval_by='uncertainty'):
    """
    Check whether the model with lower epistemic uncertainty (higher confidence)
    actually had a correct solution based on 
        1. epistemic uncertainty OR
        2. average probability.
    
    Args:
    - correct_indices: Indices where one model is correct and the other is incorrect.
    - P_LLM: Predictions from the LLM model.
    - P_MC: Predictions from the MC model.
    - sigma2_LLM: Epistemic uncertainty of the LLM model.
    - sigma2_MC: Epistemic uncertainty of the MC model.
    - ground_truths: Ground truth labels (binary, array of shape [num_samples]).
    - eval_by: Whether to evaluate based on 'uncertainty' or 'probability' (str).
    
    Returns:
    - How often the lower uncertainty model was correct (for that timestep) ito
        1. epistemic uncertainty OR
        2. average probability.
    """

    selected_and_correct = 0
    for idx in correct_indices:

        if eval_by == 'uncertainty':
            # Check which model has lower epistemic uncertainty
            if sigma2_LLM[idx] < sigma2_MC[idx]:  # LLM has lower uncertainty
                selected_prediction = P_LLM[idx]
            else:  # MC has lower uncertainty
                selected_prediction = P_MC[idx]
        
        elif eval_by == 'probability':
            # Check which model has higher average probability
            if P_LLM[idx] > P_MC[idx]:
                selected_prediction = P_LLM[idx]
            else:
                selected_prediction = P_MC[idx]
        
        # Check if selected model's prediction was correct
        if (selected_prediction >= 0.5) == ground_truths[idx]:
            selected_and_correct += 1
    
    return selected_and_correct / len(correct_indices) if len(correct_indices) > 0 else 0


def analyze_improvement(correct_llm_incorrect_mc, correct_mc_incorrect_llm, P_combined, ground_truths):
    """
    How often using the OTHER model's prediction would have improved the aggregated model
    (when only one of the models is correct).
    
    Args:
    - correct_llm_incorrect_mc: Indices where LLM is correct and MC is incorrect.
    - correct_mc_incorrect_llm: Indices where MC is correct and LLM is incorrect.
    - P_combined: Aggregated model's predictions.
    - ground_truths: Ground truth labels (binary, array of shape [num_samples]).
    
    Returns:
    - (prop where using LLM would improve agg, prop where using MC would improve agg)
    """

    improve_with_llm = 0
    improve_with_mc = 0

    # all 'one right one wrong' cases
    total_cases = len(correct_llm_incorrect_mc) + len(correct_mc_incorrect_llm)

    # LLM correct, MC incorrect
    for idx in correct_llm_incorrect_mc:
        # check if aggregated model incorrect AND if it could have been improved by LLM
        if (P_combined[idx] >= 0.5) != ground_truths[idx]: 
            improve_with_llm += 1  

    # MC correct, LLM incorrect
    for idx in correct_mc_incorrect_llm:
        # check if aggregated model incorrect AND if it could have been improved by 2-stage
        if (P_combined[idx] >= 0.5) != ground_truths[idx]:  
            improve_with_mc += 1 

    percent_improve_with_llm = (improve_with_llm / total_cases) if total_cases > 0 else 0
    percent_improve_with_mc = (improve_with_mc / total_cases) if total_cases > 0 else 0

    return percent_improve_with_llm, percent_improve_with_mc