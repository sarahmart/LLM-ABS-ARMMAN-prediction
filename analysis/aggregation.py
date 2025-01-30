# imports
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


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

    if len(ground_truths) == 0:
        print("NO GROUND TRUTHS??")
        return 0, 0, 0
    
    # Binarize predictions based on threshold
    P_combined = np.array(P_combined)
    predictions = (P_combined >= threshold).astype(int)
    
    # Compute accuracy & F1 with sklearn
    accuracy = accuracy_score(ground_truths, predictions)
    f1 = f1_score(ground_truths, predictions, zero_division=0)
    
    # Compute log likelihood (binary cross-entropy)
    epsilon = 1e-10  # To avoid log(0) errors
    P_combined_clipped = np.clip(P_combined, epsilon, 1 - epsilon)
    log_likelihood = np.sum(ground_truths * np.log(P_combined_clipped) + (1-ground_truths) * np.log(1-P_combined_clipped)) / len(ground_truths) # normalise by pop size
    
    return accuracy, f1, log_likelihood


def compute_metrics_by_group(data, predictions, ground_truths, feature_categories, models, P_combined, P_direct_avg, P_lowest_unc):
    """
    Compute metrics (accuracy, F1 score, log likelihood) by feature group, 
    including aggregated and averaged models.
    """
    metrics_by_group = {model: {} for model in models + ["Aggregated", "Averaged", "Lowest Uncertainty"]}

    # Loop over timesteps
    for t in range(ground_truths.shape[0]):  
        for category in feature_categories:
            # Filter data by feature category (boolean index for mothers)
            group_indices = data[category] == 1  # bool array of shape [mothers]
            group_ground_truths = ground_truths[t, group_indices]  

            # Skip empty groups
            if len(group_ground_truths) == 0:
                continue

            # Compute metrics for each model
            for model in models:
                # Get predictions for this timestep and group
                group_predictions = np.array(predictions[model])[t, group_indices]

                # Compute metrics
                acc, f1, log_likelihood = compute_metrics(group_predictions, group_ground_truths)

                # Store metrics
                if category not in metrics_by_group[model]:
                    metrics_by_group[model][category] = {"Accuracy": [], "F1 Score": [], "Log Likelihood": []}
                metrics_by_group[model][category]["Accuracy"].append(acc)
                metrics_by_group[model][category]["F1 Score"].append(f1)
                metrics_by_group[model][category]["Log Likelihood"].append(log_likelihood)

            # Include aggregated models
            for aggregate_type, aggregate_preds in zip(["Aggregated", "Averaged", "Lowest Uncertainty"], [P_combined, P_direct_avg, P_lowest_unc]):
                aggregate_preds = np.array(aggregate_preds)  # Convert to NumPy array
                group_aggregate_predictions = aggregate_preds[t, group_indices]
                acc, f1, log_likelihood = compute_metrics(group_aggregate_predictions, group_ground_truths)

                # Store metrics
                if category not in metrics_by_group[aggregate_type]:
                    metrics_by_group[aggregate_type][category] = {"Accuracy": [], "F1 Score": [], "Log Likelihood": []}
                metrics_by_group[aggregate_type][category]["Accuracy"].append(acc)
                metrics_by_group[aggregate_type][category]["F1 Score"].append(f1)
                metrics_by_group[aggregate_type][category]["Log Likelihood"].append(log_likelihood)

    return metrics_by_group

def overall_metrics_baselines(metric_dict, n_weeks=None):
    """
    Compute overall metrics for the baselines (average, lowest uncertainty).
    """

    if n_weeks is None:
        n_weeks = len(metric_dict["Accuracy"])
    
    overall_acc = np.mean(metric_dict["Accuracy"][:n_weeks])
    overall_f1 = np.mean(metric_dict["F1 Score"][:n_weeks])
    overall_log_lik = np.mean(metric_dict["Log Likelihood"][:n_weeks])

    return [overall_acc, overall_f1, overall_log_lik]

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

        # Add small random noise to avoid determinisism
        arm_predictions = arm_predictions + np.random.normal(0, 1e-5, size=arm_predictions.shape)
        arm_predictions = np.clip(arm_predictions, 1e-10, 1 - 1e-10)

        # Mean prediction across queries for this arm
        mean_prediction = np.mean(arm_predictions)

        # Compute predictive uncertainty (entropy of the mean prediction)
        pred_uncertainty = binary_entropy(mean_prediction)
        predictive_uncertainty.append(pred_uncertainty)

        # Compute aleatoric uncertainty (mean entropy of individual predictions)
        individual_entropies = binary_entropy(arm_predictions)
        aleatoric_uncertainty.append(np.mean(individual_entropies))

        # Epistemic uncertainty : difference between predictive and aleatoric uncertainty
        epistemic_uncertainty.append(pred_uncertainty - np.mean(individual_entropies))

    return epistemic_uncertainty, aleatoric_uncertainty, predictive_uncertainty


def infer_posterior(*predictions):
    """
    Infer the posterior probability using multiple models as priors.

    Args:
    - *predictions: A variable number of arrays, where each array represents the probability of engagement
                    predicted by a model (each array should have shape [num_samples]).

    Returns:
    - P_posterior: Posterior probability of engagement (array of shape [num_samples]).
    """

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

    return P_posterior


def uncertainty_based_selection(predictions, uncertainties, normalization_method=None):
    """
    Aggregate predictions by selecting the model for each arm and timestep with the *lowest epistemic uncertainty*.
    That lowest uncertainty value is the corresponding uncertainty for the now-aggregated prediction.

    Args:
    - predictions: List of arrays, where each array is the probability of engagement predicted by a model 
                   (shape [num_samples] for each array).
    - uncertainties: List of arrays, where each array is the epistemic uncertainty of the corresponding model's predictions 
                     (shape [num_samples] for each array).

    Returns:
    - P_combined: Combined posterior probability of engagement, selecting the lowest uncertainty for each sample 
                  (array of shape [num_samples]).
    """

    # Normalize uncertainties across models (if normalization_method)
    if normalization_method is not None and uncertainties is not None:
        uncertainties = normalization_method(uncertainties) 
    
    # Stack predictions and uncertainties into arrays of shape [num_samples, num_models]
    # i.e. each row is an arm, each col corr to a model
    predictions = np.stack(predictions, axis=1)
    uncertainties = np.stack(uncertainties, axis=1)

    # Models with the minimum uncertainty for each arm
    min_uncertainty_values = np.min(uncertainties, axis=1)

    # Deal w ties: average predictions if multiple models have same min uncertainty
    tie_masks = uncertainties == min_uncertainty_values[:, None]
    P_combined = np.sum(predictions * tie_masks, axis=1) / np.sum(tie_masks, axis=1)
    
    return P_combined


def direct_averaging(predictions):
    """
    Aggregates predictions from multiple models by averaging across predictions 
    and normalizing uncertainties across models.

    Args:
    - predictions: List of arrays, where each array is the probability of engagement predicted by a model 
                   (each array should have shape [num_samples]).

    Returns:
    - P_avg: Average probability of engagement across models (array of shape [num_samples]).
    """

    # Average predictions across models
    P_avg = np.mean(predictions, axis=0)
    P_avg = np.clip(P_avg, 0, 1)

    # Handle NaNs by replacing with 0.5 probability
    if np.any(np.isnan(P_avg)):
        print("Warning: NaN values detected in P_avg. Replacing with 0.5.")
        P_avg = np.nan_to_num(P_avg, nan=0.5)

    return P_avg


def bayesian_aggregation(predictions, uncertainties, normalization_method=None):
    """
    Uncertainty-weighted aggregation:
    Aggregates predictions from multiple models using Bayesian weighting with a linear combination.

    Args:
    - predictions: List of arrays, where each array is the probability of engagement predicted by a model 
                   (each array should have shape [num_samples]).
    - uncertainties: List of arrays, where each array is the epistemic uncertainty of the corresponding model's predictions 
                     (each array should have shape [num_samples]).
    - normalization_method: Function to normalize uncertainties from normalization.py.

    Returns:
    - P_combined: Combined posterior probability of engagement (array of shape [num_samples]).
    """

    # Normalize uncertainties across models (if normalization_method)
    if normalization_method is not None and uncertainties is not None:
        uncertainties = normalization_method(uncertainties) 
    
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
    
    return P_combined


# def identify_discrepancies(P_LLM, P_MC, ground_truths, threshold=0.5):
#     """
#     Identify indices (arms) where one model is correct and the other is incorrect.
    
#     Args:
#     - P_LLM: Predictions from the LLM model.
#     - P_MC: Predictions from the MC model.
#     - ground_truths: Ground truth labels.
#     - threshold: Threshold to classify predictions as engaged or not engaged.

#     Returns:
#     - correct_llm_incorrect_mc: Indices where LLM is correct and MC is incorrect.
#     - correct_mc_incorrect_llm: Indices where MC is correct and LLM is incorrect.
#     """
#     predictions_llm = (P_LLM >= threshold).astype(int)
#     predictions_mc = (P_MC >= threshold).astype(int)
    
#     correct_llm = (predictions_llm == ground_truths)
#     correct_mc = (predictions_mc == ground_truths)
    
#     correct_llm_incorrect_mc = np.where((correct_llm == 1) & (correct_mc == 0))[0]
#     correct_mc_incorrect_llm = np.where((correct_mc == 1) & (correct_llm == 0))[0]
    
#     return correct_llm_incorrect_mc, correct_mc_incorrect_llm


# def compare_confidence(correct_indices, P_LLM, P_MC, sigma2_LLM, sigma2_MC, ground_truths, eval_by='uncertainty'):
#     """
#     Check whether the model with lower epistemic uncertainty (higher confidence)
#     actually had a correct solution based on 
#         1. epistemic uncertainty OR
#         2. average probability.
    
#     Args:
#     - correct_indices: Indices where one model is correct and the other is incorrect.
#     - P_LLM: Predictions from the LLM model.
#     - P_MC: Predictions from the MC model.
#     - sigma2_LLM: Epistemic uncertainty of the LLM model.
#     - sigma2_MC: Epistemic uncertainty of the MC model.
#     - ground_truths: Ground truth labels (binary, array of shape [num_samples]).
#     - eval_by: Whether to evaluate based on 'uncertainty' or 'probability' (str).
    
#     Returns:
#     - How often the lower uncertainty model was correct (for that timestep) ito
#         1. epistemic uncertainty OR
#         2. average probability.
#     """

#     selected_and_correct = 0
#     for idx in correct_indices:

#         if eval_by == 'uncertainty':
#             # Check which model has lower epistemic uncertainty
#             if sigma2_LLM[idx] < sigma2_MC[idx]:  # LLM has lower uncertainty
#                 selected_prediction = P_LLM[idx]
#             else:  # MC has lower uncertainty
#                 selected_prediction = P_MC[idx]
        
#         elif eval_by == 'probability':
#             # Check which model has higher average probability
#             if P_LLM[idx] > P_MC[idx]:
#                 selected_prediction = P_LLM[idx]
#             else:
#                 selected_prediction = P_MC[idx]
        
#         # Check if selected model's prediction was correct
#         if (selected_prediction >= 0.5) == ground_truths[idx]:
#             selected_and_correct += 1
    
#     return selected_and_correct / len(correct_indices) if len(correct_indices) > 0 else 0


# def analyze_improvement(correct_llm_incorrect_mc, correct_mc_incorrect_llm, P_combined, ground_truths):
#     """
#     How often using the OTHER model's prediction would have improved the aggregated model
#     (when only one of the models is correct).
    
#     Args:
#     - correct_llm_incorrect_mc: Indices where LLM is correct and MC is incorrect.
#     - correct_mc_incorrect_llm: Indices where MC is correct and LLM is incorrect.
#     - P_combined: Aggregated model's predictions.
#     - ground_truths: Ground truth labels (binary, array of shape [num_samples]).
    
#     Returns:
#     - (prop where using LLM would improve agg, prop where using MC would improve agg)
#     """

#     improve_with_llm = 0
#     improve_with_mc = 0

#     # all 'one right one wrong' cases
#     total_cases = len(correct_llm_incorrect_mc) + len(correct_mc_incorrect_llm)

#     # LLM correct, MC incorrect
#     for idx in correct_llm_incorrect_mc:
#         # check if aggregated model incorrect AND if it could have been improved by LLM
#         if (P_combined[idx] >= 0.5) != ground_truths[idx]: 
#             improve_with_llm += 1  

#     # MC correct, LLM incorrect
#     for idx in correct_mc_incorrect_llm:
#         # check if aggregated model incorrect AND if it could have been improved by 2-stage
#         if (P_combined[idx] >= 0.5) != ground_truths[idx]:  
#             improve_with_mc += 1 

#     percent_improve_with_llm = (improve_with_llm / total_cases) if total_cases > 0 else 0
#     percent_improve_with_mc = (improve_with_mc / total_cases) if total_cases > 0 else 0

#     return percent_improve_with_llm, percent_improve_with_mc