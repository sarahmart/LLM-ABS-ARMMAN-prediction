import numpy as np
from sklearn.metrics import f1_score


# Function to compute MSE
def compute_mse(predictions, ground_truths):
    return np.mean((np.array(predictions) - np.array(ground_truths)) ** 2)


# Function to compute accuracy using majority voting
def compute_accuracy(predictions, ground_truths):
    predicted_engagement = np.array(predictions) > 0.5
    actual_engagement = np.array(ground_truths) > 30
    accuracy = np.mean(predicted_engagement == actual_engagement)
    return accuracy


# Function to compute log likelihood
def compute_log_likelihood(predictions, ground_truths):
    actual_engagement = np.array(ground_truths) > 30
    actual_engagement = actual_engagement.astype(float)

    predicted_probs = np.clip(predictions, 1e-10, 1-1e-10)  # Avoid log(0)
    log_likelihood = np.mean(actual_engagement * np.log(predicted_probs) + 
                            (1 - actual_engagement) * np.log(1 - predicted_probs))
    return log_likelihood

# Function to compute log likelihood for binary predictions
def compute_log_likelihood_bin(predicted_binary, ground_truths):
    actual_engagement = np.array(ground_truths) > 30
    predicted_probs = np.clip(predicted_binary, 1e-10, 1-1e-10)  # Avoid log(0)
    log_likelihood = np.mean(actual_engagement * np.log(predicted_probs) + 
                            (1 - actual_engagement) * np.log(1 - predicted_probs))
    return log_likelihood


# Function to compute total accuracy, F1 score, and log likelihood
def compute_total_metrics(all_binary_predictions, all_ground_truths, t1, t2):
    flat_predictions = []
    flat_ground_truths = []
    
    for m in range(t1, t2+1):
        flat_predictions.extend(all_binary_predictions[m])
        flat_ground_truths.extend(all_ground_truths[m])
    
    # Convert lists to numpy arrays for metric computation
    flat_predictions = np.array(flat_predictions)
    flat_ground_truths = np.array(flat_ground_truths)

    # Compute total accuracy and F1 score
    binary_predictions = flat_predictions > 0.5
    binary_ground_truths = flat_ground_truths 
    total_accuracy = np.mean(binary_predictions == binary_ground_truths)
    total_f1_score = f1_score(binary_ground_truths, binary_predictions, zero_division=1)
    
    # Compute total log likelihood
    total_log_likelihood = compute_log_likelihood_bin(flat_predictions, flat_ground_truths)

    return total_accuracy, total_f1_score, total_log_likelihood