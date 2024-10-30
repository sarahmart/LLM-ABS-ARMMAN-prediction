import numpy as np


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