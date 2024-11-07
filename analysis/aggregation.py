import numpy as np
from scipy.stats import entropy
import json
from sklearn.metrics import accuracy_score, f1_score

# from aggregation_all import plot_uncertainties_vs_month, plot_distribution_over_time, analyze_improvement, compare_confidence, identify_discrepancies
# from normalization import *

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
    
    # Binarize the predictions based on the threshold and ensure they are integers
    predictions = (P_combined >= threshold).astype(int)

    # Check if predictions and ground_truths are binary (0 or 1)
    print(f"Unique values in predictions: {np.unique(predictions)}")
    print(f"Unique values in ground_truths: {np.unique(ground_truths)}")
    
    # Compute accuracy
    accuracy = accuracy_score(ground_truths, predictions)
    
    # Compute F1 score
    f1 = f1_score(ground_truths, predictions)
    
    # Compute log likelihood (binary cross-entropy)
    epsilon = 1e-10  # To avoid log(0) errors
    P_combined_clipped = np.clip(P_combined, epsilon, 1 - epsilon)
    log_likelihood = np.sum(
        ground_truths * np.log(P_combined_clipped) + (1 - ground_truths) * np.log(1 - P_combined_clipped)
    )
    
    return accuracy, f1, log_likelihood


def binary_entropy(p):
    """Compute binary entropy for Bernoulli distribution."""
    p = np.clip(p, 1e-10, 1 - 1e-10)  # Avoid log(0) errors
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def load_predictions_and_ground_truths(predictions_path, ground_truths_path):
    """Load saved individual predictions and ground truths from JSON files."""
    with open(predictions_path, 'r') as f:
        all_individual_predictions = json.load(f)
    
    with open(ground_truths_path, 'r') as f:
        ground_truths = json.load(f)
    
    return all_individual_predictions, ground_truths


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


def infer_posterior(P_data, P_LLM):
    """
    Infer the posterior probability using LLM as a prior.

    Args:
    - P_data: Probability of engagement predicted by the data-driven model (array of shape [num_samples]).
    - P_LLM: Probability of engagement predicted by the LLM (array of shape [num_samples]).

    Returns:
    - P_posterior: Posterior probability of engagement (array of shape [num_samples]).
    """
    # Compute probabilities of not engaging for both data and LLM
    P_data_not_engage = 1 - P_data
    P_LLM_not_engage = 1 - P_LLM

    # Apply the Bayesian formula
    numerator = P_data * P_LLM
    denominator = (P_data * P_LLM) + (P_data_not_engage * P_LLM_not_engage)
    
    # Avoid division by zero in case of very small values
    epsilon = 1e-10
    P_posterior = numerator / (denominator + epsilon)

    return P_posterior


def uncertainty_based_selection(P_LLM, P_NN, sigma2_LLM, sigma2_NN):
    """
    Aggregate the predictions by selecting the model with lower epistemic uncertainty.

    Args:
    - P_LLM: Probability of engagement predicted by the LLM (array of shape [num_samples]).
    - P_NN: Probability of engagement predicted by the data-driven model (array of shape [num_samples]).
    - sigma2_LLM: Epistemic uncertainty of the LLM predictions (array of shape [num_samples]).
    - sigma2_NN: Epistemic uncertainty of the NN predictions (array of shape [num_samples]).
    
    Returns:
    - P_combined: Combined posterior probability of engagement (array of shape [num_samples]).
    """
    # Select the model with the lower epistemic uncertainty for each test point
    P_combined = np.where(sigma2_LLM < sigma2_NN, P_LLM, P_NN)
    
    return P_combined


def bayesian_aggregation(P_LLM, P_data, sigma2_LLM, sigma2_data, normalization_method=None):
    """
    Aggregate the predictions from LLM and data-driven model using Bayesian weighting with a linear combination.
    
    Args:
    - P_LLM: Probability of engagement predicted by the LLM (array of shape [num_samples]).
    - P_data: Probability of engagement predicted by the data-driven model (array of shape [num_samples]).
    - sigma2_LLM: Epistemic uncertainty of the LLM predictions (array of shape [num_samples]).
    - sigma2_data: Epistemic uncertainty of the data-driven model predictions (array of shape [num_samples]).
    - normalization_method: Function to normalize uncertainties from normalization.py.

    
    Returns:
    - P_combined: Combined posterior probability of engagement (array of shape [num_samples]).
    """

    if normalization_method is not None:
        sigma2_LLM = normalization_method(sigma2_LLM)
        sigma2_data = normalization_method(sigma2_data)

    # Add a small epsilon to avoid division by zero or near-zero uncertainties
    epsilon = 1e-10
    sigma2_LLM = np.array(sigma2_LLM) + epsilon
    sigma2_data = np.array(sigma2_data) + epsilon
    
    # Convert epistemic uncertainties to precision (tau)
    tau_LLM = 1 / sigma2_LLM
    tau_data = 1 / sigma2_data
    
    # Compute alpha for each test point (the weight for LLM)
    alpha = tau_LLM / (tau_LLM + tau_data)
    
    # Combine the predictions as a weighted linear combination
    P_combined = alpha * P_LLM + (1 - alpha) * P_data
    
    # Ensure the combined probabilities stay within [0, 1]
    P_combined = np.clip(P_combined, 0, 1)
    
    ## how many NaN values are in P_combined
    print(f"Number of NaN values in P_combined: {np.sum(np.isnan(P_combined))}")
    # Check for NaN values and handle them
    if np.any(np.isnan(P_combined)):
        print("Warning: NaN values detected in P_combined. Replacing with 0.5.")
        P_combined = np.nan_to_num(P_combined, nan=0.5)  # Replace NaN with neutral probability (0.5)
    
    return P_combined


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