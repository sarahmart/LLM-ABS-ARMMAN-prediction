import json

def load_predictions_and_ground_truths(predictions_path, ground_truths_path):
    """Load saved individual predictions and ground truths from JSON files."""
    with open(predictions_path, 'r') as f:
        all_individual_predictions = json.load(f)
    
    with open(ground_truths_path, 'r') as f:
        ground_truths = json.load(f)
    
    return all_individual_predictions, ground_truths


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