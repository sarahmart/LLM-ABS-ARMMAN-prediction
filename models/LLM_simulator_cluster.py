import json
import os
import requests
import time

import numpy as np

import vertexai
from vertexai.generative_models import GenerativeModel

from datetime import datetime
from requests.exceptions import Timeout
from sklearn.metrics import f1_score
from tqdm import tqdm

from preprocess import data_preprocessing, map_features_to_prompt
from prompt_templates import *
from metrics import *


# TODO:
## onboarding google doc for server access --> faster results
## show no difference between 4o-mini and 4o as justification to use mini 


def load_config(config_path):
    """Load the configuration file for the LLM API."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def LLM_eval(config: dict, model: str, sys_prompt: str, user_prompt: str, max_retries: int = 10) -> int:
    """Evaluate a prompt using the LLM model with retry logic for incorrect format."""
    
    # Model-dependent headers for the API request
    if "openai" in model:
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

    elif "anthropic" in model:
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
                             "text": "Answer the following question concisely with either '##Yes##' OR '##No##' and avoid extra details." +
                             sys_prompt + user_prompt
                            }
                        ]
                    }
                ],
                "temperature": 0.7,
            }
        }

    elif "meta" in model:
        headers = {
        "x-api-key": config["api_key"],
        "Content-Type": "application/json",
        "Accept": "application/json"
        }
        data = {
        "modelId": "meta.llama3-70b-instruct-v1:0",
        "body": {
            "prompt": f"""
                Answer the following question concisely in English (no code) with either '##Yes##' OR '##No##' and avoid extra details.
                {sys_prompt + user_prompt}
            """,
            "temperature": 0.7, 
            # "top_p": 0.5            
        }
    }
    
    elif "google" in model:
        PROJECT_ID = "seas-dev-llmsimulator-4322"
        vertexai.init(project=PROJECT_ID, location="us-central1")
        model = GenerativeModel(config["model"])
    
    else:
        raise ValueError("Specified model not supported.")

    for attempt in range(max_retries):
        try:
            if "google" in model:
                response = model.generate_content(sys_prompt + user_prompt)
            else:
                response = requests.post(config["api_url"], headers=headers, data=json.dumps(data), timeout=120)

            # Extract prediction
            if response.status_code == 200:
                response_data = response.json()

                if "openai" in model:
                    prediction = response_data["choices"][0]["message"]["content"]
                elif "anthropic" in model:
                    prediction = response_data["content"][0]["text"]
                elif "meta" in model:
                    prediction = response_data["generation"]
                elif "google" in model:
                    prediction = response.text

                if "#Yes#" in prediction and "#No#" in prediction:
                    raise ValueError("Prediction not consistent.")
                elif "#Yes#" in prediction:
                    return 1, prediction  # Engaged
                elif "#No#" in prediction:
                    return 0, prediction   # Not Engaged
                else:
                    raise ValueError("Prediction does not follow the expected format.")
            
            else:
                    print(f"API request failed with status {response.status_code}: {response.text}")
                    time.sleep(2) 
                    continue
            
        except ValueError as ex:
            print(f"Extraction failed on attempt {attempt + 1}: {ex}")
            if attempt < max_retries - 1:
                print("Retrying...")
                time.sleep(2)

        except Timeout:
            print(f"Request timed out after 2 mins.")
            if attempt < max_retries - 1:
                print("Retrying...")
                time.sleep(2)
        
        except Exception as ex:
            print(ex)
            time.sleep(3)
            continue  

    # may not need to return None here, check 
    return "error", None  # Return error if all attempts fail


def generate_prompt(mapped_features, prompt_template):
    """Generate a prompt using the provided template, mapped features, and current time step t."""
    # If t is 0, it is the starting week, so adjust the prompt accordingly
    return prompt_template.format(**mapped_features)


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

    with open(f"./results/{args.config_path.split('_')[0]}_{args.num_arms}_{args.t1}_{args.t2}/engagement_structured_prompts_and_responses_{args.prompt_version}_filter_{str(args.filter)}.json", "w") as json_file:
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


def process_data_monthly_with_prompt_ensemble(args, config, features, state_trajectories, action_trajectories, 
                                              sys_prompt, num_queries=5):
    # Use all prompt versions in the ensemble
    prompt_templates = [bin_prompt_v1(), bin_prompt_v2(), bin_prompt_v3(), bin_prompt_v4(), bin_prompt_v5()] 
    starting_prompt_templates = [starting_prompt_v2(), starting_prompt_v3(), starting_prompt_v4(), starting_prompt_v5(), starting_prompt_v6()]
    
    month_steps = [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32), (32, 36), (36, 39)] # specify which weeks corresopnd to which month
    
    all_binary_predictions = [[] for _ in range(len(month_steps))]
    all_ground_truths = [[] for _ in range(len(month_steps))]
    all_individual_predictions = []  # Store all individual predictions
    extraction_failures = 0
    structured_results = {}
    ground_truths = []

    # Loop through each time step, but only use LLM between t1 and t2
    for m, (start, end) in tqdm(enumerate(month_steps), desc="Processing months"):

        if m < args.t1: # skip LLM predictions
            continue  # Skip LLM predictions for months before t1

        # From month t1 to t2, use AR LLM predictions
        if args.t1 <= m <= args.t2:

            structured_results[m] = {}

            for arm in tqdm(range(args.num_arms), desc="Processing arms", leave=False):
                structured_results[m][arm] = {"prompt": [], "responses": []}
                
                arm_features = features[arm]

                start_week_next_month = month_steps[m][0]
                end_week_next_month = month_steps[m][1]

                if m == 0:
                    mapped_features = map_features_to_prompt(arm_features, [], [], first_week=True)
                    prompt_text = [generate_prompt(mapped_features, prompt_template) for prompt_template in starting_prompt_templates]
                else:
                    history_end_week = month_steps[m-1][1]
                    arm_states = state_trajectories[arm][:history_end_week]
                    arm_actions = action_trajectories[arm][:history_end_week]
                    
                    mapped_features = map_features_to_prompt(arm_features, arm_states, arm_actions)
                    prompt_text = [generate_prompt(mapped_features, prompt_template) for prompt_template in prompt_templates]

                ground_truth = 1 if np.sum(np.array(state_trajectories[arm][start_week_next_month:end_week_next_month]) > 30) > 0 else 0
                
                ground_truths.append(ground_truth)
                predictions = []
                individual_predictions = []  # Store individual predictions for this arm and month
                
                for prompt in prompt_text:
                    responses = []
                    for _ in range(num_queries):

                        engagement_prediction, response = LLM_eval(config, args.config_path.split('_')[0], sys_prompt, prompt)

                        if engagement_prediction == "error":
                            extraction_failures += 1
                            engagement_prediction = 0
                        responses.append(engagement_prediction)
                    
                    structured_results[m][arm]["responses"].append(response)
                    predictions.append(np.mean(responses))    # Average responses for this prompt template
                    individual_predictions.append(responses)  # Save individual predictions for each prompt

                # Ensemble: Majority voting or averaging across prompt templates
                final_engagement_prediction = np.mean(predictions)
                all_binary_predictions[m].append(final_engagement_prediction)
                all_ground_truths[m].append(ground_truth)
                all_individual_predictions.append(individual_predictions)  # Save all individual predictions


    model = args.config_path.split('_')[0]
    os.makedirs(f"./results/montly/{model}_{args.num_arms}", exist_ok=True)

    # Save the individual predictions to JSON
    with open(f"./results/monthly/{model}_{args.num_arms}/all_individual_predictions_t1_{args.t1}_t2_{args.t2}.json", "w") as json_file:
        json.dump(all_individual_predictions, json_file, indent=4)
    
    ## Save ground truths
    with open(f"./results/monthly/{model}_{args.num_arms}/ground_truths_t1_{args.t1}_t2_{args.t2}.json", "w") as json_file:
        json.dump(ground_truths, json_file, indent=4)

    # Save structured_results as JSON
    with open(f"./results/monthly/{model}_{args.num_arms}/structured_results_t1_{args.t1}_t2_{args.t2}.json", 'w') as f:
        json.dump(structured_results, f, indent=4)

    # Save results and return
    return (all_ground_truths, all_binary_predictions, extraction_failures)


def process_data_weekly_with_prompt_ensemble(args, config, features, state_trajectories, action_trajectories, 
                                              sys_prompt, num_queries=5):
    # Use all prompt versions in the ensemble
    prompt_templates = [bin_prompt_v1(), bin_prompt_v2(), bin_prompt_v3(), bin_prompt_v4(), bin_prompt_v5()] 
    starting_prompt_templates = [starting_prompt_v2(), starting_prompt_v3(), starting_prompt_v4(), starting_prompt_v5(), starting_prompt_v6()]

    week_steps = np.arange(40)

    all_binary_predictions = [[] for _ in range(args.t2)]  # To store binary predictions for t2 steps
    all_ground_truths = [[] for _ in range(args.t2)]
    all_individual_predictions = []  # Store all individual predictions
    extraction_failures = 0
    structured_results = {}
    ground_truths = []

    for w in tqdm(range(args.t2 - args.t1), desc="Processing steps", leave=False):
        
        if w < args.t1: # skip LLM predictions
            continue  # Skip LLM predictions for months before t1

        # From month t1 to t2, use AR LLM predictions
        if args.t1 <= w <= args.t2:
            
            structured_results[w] = {}

            for arm in tqdm(range(args.num_arms), desc="Processing arms", leave=False):
                structured_results[w][arm] = {"prompt": [], "responses": []}
                
                arm_features = features[arm]

                if w == 0:
                    mapped_features = map_features_to_prompt(arm_features, [], [], first_week=True)
                    prompt_text = [generate_prompt(mapped_features, prompt_template) for prompt_template in starting_prompt_templates]
                else:
                    history_end_week = week_steps[w-1]
                    arm_states = state_trajectories[arm][:history_end_week]
                    arm_actions = action_trajectories[arm][:history_end_week]
                    
                    mapped_features = map_features_to_prompt(arm_features, arm_states, arm_actions)
                    prompt_text = [generate_prompt(mapped_features, prompt_template) for prompt_template in prompt_templates]

                ground_truth = 1 if np.array(state_trajectories[arm][w]) > 30 else 0

                ground_truths.append(ground_truth)
                predictions = []
                individual_predictions = []  # Store individual predictions for this arm and week
                
                for prompt in prompt_text:
                    responses = []
                    for _ in range(num_queries):
                        engagement_prediction, response = LLM_eval(config, args.config_path.split('_')[0], sys_prompt, prompt)
                        if engagement_prediction == "error":
                            extraction_failures += 1
                            engagement_prediction = 0
                        responses.append(engagement_prediction)
                    structured_results[w][arm]["responses"].append(response)
                    predictions.append(np.mean(responses))    # Average responses for this prompt template
                    individual_predictions.append(responses)  # Save individual predictions for each prompt

                # Ensemble: Majority voting or averaging across prompt templates
                final_engagement_prediction = np.mean(predictions)
                all_binary_predictions[w].append(final_engagement_prediction)
                all_ground_truths[w].append(ground_truth)
                all_individual_predictions.append(individual_predictions)  # Save all individual predictions


    model = args.config_path.split('_')[0]
    os.makedirs(f"./results/weekly/{model}_{args.num_arms}", exist_ok=True)

    # Save the individual predictions to JSON
    with open(f"./results/weekly/{model}_{args.num_arms}/all_individual_predictions_t1_{args.t1}_t2_{args.t2}.json", "w") as json_file:
        json.dump(all_individual_predictions, json_file, indent=4)
    
    ## Save ground truths
    with open(f"./results/weekly/{model}_{args.num_arms}/ground_truths_t1_{args.t1}_t2_{args.t2}.json", "w") as json_file:
        json.dump(ground_truths, json_file, indent=4)

    # Save structured_results as JSON
    with open(f"./results/weekly/{model}_{args.num_arms}/structured_results_t1_{args.t1}_t2_{args.t2}.json", 'w') as f:
        json.dump(structured_results, f, indent=4)

    # Save results and return
    return (all_ground_truths, all_binary_predictions, extraction_failures)