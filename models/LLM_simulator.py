import json

import numpy as np

from datetime import datetime

from preprocess import data_preprocessing, map_features_to_prompt
from prompt_templates import *
from metrics import *

####

import time
import requests
import re
from scipy.special import logit, expit
from tqdm import tqdm  # Import tqdm for progress bars
from sklearn.metrics import f1_score
####

# TODO:
## autoregressive predictions
## binary classification 
## onboarding google doc for server access --> faster results
## show no difference between 4o-mini and 4o as justification to use mini 


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


def load_config(config_path):
    """Load the configuration file for the LLM API."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


# def extract_listening_time(prediction):
#     match = re.search(r"Predicted Listening Time: (\d+) seconds", prediction)
#     if match:
#         return int(match.group(1))
#     else:
#         raise ValueError("Prediction does not follow the expected format.")

# def gpt4_eval(config: dict, sys_prompt: str, user_prompt: str, max_retries: int = 10) -> int:
#     """Evaluate a prompt using the LLM model with retry logic for incorrect format."""
    
#     # Set the API key from the config using requests.post
#     headers = {
#         "Content-Type": "application/json",
#         "api-key": config["api_key"]
#     }

#     data = {
#         "model": config["model"],
#         "messages": [
#             {"role": "system", "content": sys_prompt},
#             {"role": "user", "content": user_prompt}
#         ],
#         "temperature": 0.7,
#         "max_tokens": 2048
#     }

#     for attempt in range(max_retries):
#         try:
#             # Using OpenAI's Python client, but with flexibility from the config
#             response = requests.post(config["api_url"], headers=headers, data=json.dumps(data))

#             # Extracting the prediction
#             if response.status_code == 200:
#                 response_data = response.json()
#                 prediction = response_data["choices"][0]["message"]["content"]

#                 listening_time = extract_listening_time(prediction)
#                 return listening_time, prediction
            
#             else:
#                 print(f"API request failed with status {response.status_code}: {response.text}")
#                 time.sleep(2)  # Retry delay
#                 continue

#         except Exception as ex:
#             print(f"Extraction failed on attempt {attempt + 1}: {ex}")
#             time.sleep(3)
#             continue

#     return "error", None  # Return an error message if all attempts fail


def LLM_eval(config: dict, model: str, sys_prompt: str, user_prompt: str, max_retries: int = 10) -> int:
    """Evaluate a prompt using the LLM model with retry logic for incorrect format."""
    
    # Model-dependent headers for the API request
    if model == "openai":
        headers = {
            "Content-Type": "application/json",
            "api-key": config["api_key"]
        }
        data = {
        "model": config["model"], 
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 2048
        }

    elif model == "anthropic":
        headers = {
            "x-api-key": config["api_key"],
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        data = {
            "modelId": config["model"],
            "body": {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 2048,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text",
                             "text": sys_prompt + user_prompt
                            }
                        ]
                    }
                ],
                "temperature": 0.7,
            }
        }
    
    else:
        raise ValueError("Specified model not supported.")

    for attempt in range(max_retries):
        try:
            response = requests.post(config["api_url"], headers=headers, data=json.dumps(data))

            # Extracting the prediction
            if response.status_code == 200:
                response_data = response.json()
                # print(response_data)

                if model == "openai":
                    prediction = response_data["choices"][0]["message"]["content"]
                elif model == "anthropic":
                    prediction = response_data["content"][0]["text"]

                print(prediction)

                if "#Yes#" in prediction:
                    return 1, prediction  # Engaged
                elif "#No#" in prediction:
                    return 0, prediction   # Not Engaged
                else:
                    raise ValueError("Prediction does not follow the expected format.")
            
        except ValueError as ex:
            print(f"Extraction failed on attempt {attempt + 1}: {ex}")
            if attempt < max_retries - 1:
                print("Retrying...")
                time.sleep(2)

        except Exception as ex:
            print(ex)
            time.sleep(3)
            continue  # Retry in case of other errors
    
    return "error"  # Return an error message if all attempts fail


def generate_prompt(mapped_features, prompt_template, t, args):
    """Generate a prompt using the provided template, mapped features, and current time step t."""
    
    # If t is 0, it's the starting week, so adjust the prompt accordingly
    if t == 0:
        if args.use_features:
            starting_prompt = starting_prompt_v1()
        else: # Use the template without features
            starting_prompt = starting_prompt_v2()

        return starting_prompt.format(**mapped_features)
    
    else:
        # Use the regular template for time steps greater than 0
        return prompt_template.format(**mapped_features)


# def logistic_growth(t, initial_mothers, L, k, t0):
#     """Adjusted logistic growth model that starts with the initial number of mothers."""
#     # Logistic growth model adds to the initial number of mothers
#     return initial_mothers + (L - initial_mothers) / (1 + np.exp(-k * (t - t0)))


def process_data(args, engine, features, state_trajectories, action_trajectories, prompt_template, sys_prompt, num_queries=5):
    """Process the data and compute errors for the autoregressive prediction model."""

    all_binary_predictions = [[] for _ in range(args.t2)]  # To store binary predictions for t2 steps
    all_ground_truths = [[] for _ in range(args.t2)]
    accuracy_per_step = []
    log_likelihood_per_step = []
    extraction_failures = 0  # Counter for failed extractions
    f1_score_per_step = []

    structured_results = {}

    for arm in tqdm(range(args.num_arms), desc="Processing arms"):
        structured_results[arm] = {}
        arm_features = features[arm]

        # Get initial t1 time steps from the data
        arm_state = state_trajectories[arm][:args.t1]  # Use the first t1 steps as input
        arm_action = action_trajectories[arm][:args.t1]

        # Start autoregressive prediction for t2 future steps
        for t in tqdm(range(args.t2), desc="Predicting future steps", leave=False):
            # Use index `t` for structured results (not `t + args.t1`)
            structured_results[arm][t] = {
                "prompt": None,
                "responses": []
            }

            # Generate the input prompt with current state and action history
            mapped_features = map_features_to_prompt(arm_features, arm_state, arm_action)
            prompt = generate_prompt(mapped_features, prompt_template, args.t1 + t, args)
            structured_results[arm][t]["prompt"] = prompt

            predictions = []

            # Query the LLM and predict the next time step's engagement
            for _ in range(num_queries):
                engagement_prediction, response = LLM_eval(engine, args.config_path.split('_')[0], sys_prompt, prompt)

                if engagement_prediction == "error":
                    extraction_failures += 1
                    engagement_prediction = 0  # Default to Not Engaged if extraction fails

                structured_results[arm][t]["responses"].append(response)
                predictions.append(engagement_prediction)
            
            #print(predictions)
            # Use majority voting or averaging for the final engagement prediction
            final_engagement_prediction = np.mean(predictions)
            #print(final_engagement_prediction)
            # Append the LLM's prediction to the state for future prediction
            arm_state.append(50 if final_engagement_prediction > 0.5 else 0)  # Use the predicted state (engagement) as input for the next time step, 50 will considered as engaged
            #print(arm_state)
            arm_action.append(action_trajectories[arm][t + args.t1])  # Keep the action sequence unchanged

            all_binary_predictions[t].append(final_engagement_prediction)
            ground_truth = state_trajectories[arm][t + args.t1]  # The ground truth engagement for the next step
            all_ground_truths[t].append(ground_truth)

    with open(f"./autoregressive_{args.t1}_{args.t2}/engagement_structured_prompts_and_responses_{args.prompt_version}_filter_{str(args.filter)}_{timestamp}.json", "w") as json_file:
        json.dump(structured_results, json_file, indent=4)

    for t in range(args.t2):
        # Compute MSE, accuracy, log likelihood, and AUROC for each time step across all arms
        accuracy_step = compute_accuracy(all_binary_predictions[t], all_ground_truths[t])
        log_likelihood_step = compute_log_likelihood(all_binary_predictions[t], all_ground_truths[t])
        f1_score_step = f1_score((np.array(all_ground_truths[t])>30).astype(int), (np.array(all_binary_predictions[t]) > 0.5).astype(int))

        accuracy_per_step.append(accuracy_step)
        log_likelihood_per_step.append(log_likelihood_step)
        f1_score_per_step.append(f1_score_step)

        print(f"Accuracy for time step {t + args.t1}: {accuracy_step}")
        print(f"Log Likelihood for time step {t + args.t1}: {log_likelihood_step}")
        print(f"F1 Score for time step {t}: {f1_score_step}")

    # Compute total metrics across all steps and arms
    total_ground_truths = np.array(all_ground_truths).flatten()
    total_binary_predictions = np.array(all_binary_predictions).flatten()

    total_accuracy = compute_accuracy(total_binary_predictions, total_ground_truths)
    total_log_likelihood = compute_log_likelihood(total_binary_predictions, total_ground_truths)

    total_f1_score = f1_score(((total_ground_truths)>30).astype(int), (total_binary_predictions > 0.5).astype(int))

    return (all_ground_truths, all_binary_predictions, accuracy_per_step, log_likelihood_per_step, f1_score_per_step, total_accuracy, total_log_likelihood, total_f1_score, extraction_failures)