import glob
import json
import os, sys
import random
import requests
import time

import numpy as np

import vertexai
from vertexai.generative_models import GenerativeModel

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from requests.exceptions import Timeout
from requests.adapters import HTTPAdapter
from sklearn.metrics import f1_score
from tqdm import tqdm
from urllib3.util.retry import Retry

from preprocess import data_preprocessing, map_features_to_prompt
from prompt_templates import *
from metrics import *


def load_config(config_path):
    """Load the configuration file for the LLM API."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def LLM_eval(config: dict, model: str, sys_prompt: str, user_prompt: str, max_retries: int=10):
    """Evaluate a prompt using the LLM model with retry logic and exponential backoff (needed for Anthropic models)."""
    
    # Model-dependent headers for API requests

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
            }
        }
    
    elif "google" in model:
        PROJECT_ID = config["project_id"]
        vertexai.init(project=PROJECT_ID, location="us-central1")
        google_model = GenerativeModel(config["model"])
    
    else:
        raise ValueError("Specified model not yet supported.")

    # Exponential backoff --> needed for Anthropic models
    initial_backoff = 2  # Start with 2 seconds 
    max_backoff = 120    # Cap backoff time at 2 mins (??)
    connection_timeout = 30  # Connection timeout 
    read_timeout = 120       # Read timeout
    
    session = requests.Session()
    # Configure retry strategy session
    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    for attempt in range(max_retries):
        try:
            if "google" in model:
                response = google_model.generate_content(sys_prompt + user_prompt)
                prediction = response.text
            else:
                response = session.post(
                    config["api_url"],
                    headers=headers,
                    json=data, 
                    timeout=(connection_timeout, read_timeout)
                )
                
                # Handle rate limiting
                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", initial_backoff))
                    print(f"Rate limit hit. Retrying after {retry_after} seconds.")
                    time.sleep(retry_after)
                    continue
                
                # Raise for any other bad status
                response.raise_for_status()
                
                # Extract prediction based on model type
                response_data = response.json()
                if "openai" in model:
                    prediction = response_data["choices"][0]["message"]["content"]
                elif "anthropic" in model:
                    prediction = response_data["content"][0]["text"]
                elif "meta" in model:
                    prediction = response_data["generation"]
                
            # Validate prediction
            if "#Yes#" in prediction and "#No#" in prediction:
                raise ValueError("Prediction not consistent.")
            elif "#Yes#" in prediction:
                return 1, prediction
            elif "#No#" in prediction:
                return 0, prediction
            else:
                raise ValueError("Prediction does not follow the expected format.")
                
        except requests.exceptions.ConnectionError as e:
            print(f"Connection error on attempt {attempt + 1}: {e}")
            if attempt >= max_retries - 1:
                return "error", None
            # longer backoff for connection issues
            time.sleep(min(initial_backoff * (2 ** attempt), max_backoff))
            
        except requests.exceptions.Timeout as e:
            print(f"Timeout error on attempt {attempt + 1}: {e}")
            if attempt >= max_retries - 1:
                return "error", None
            time.sleep(min(initial_backoff * (2 ** attempt), max_backoff))
            
        except requests.exceptions.RequestException as e:
            print(f"Request error on attempt {attempt + 1}: {e}")
            if attempt >= max_retries - 1:
                return "error", None
            time.sleep(min(initial_backoff * (2 ** attempt), max_backoff))
            
        except ValueError as e:
            print(f"Validation error on attempt {attempt + 1}: {e}")
            if attempt >= max_retries - 1:
                return "error", None
            time.sleep(initial_backoff)
            
        except Exception as e:
            print(f"Unexpected error on attempt {attempt + 1}: {e}")
            if attempt >= max_retries - 1:
                return "error", None
            time.sleep(min(initial_backoff * (2 ** attempt), max_backoff))

    return "error", None


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
            # Average for final engagement prediction
            final_engagement_prediction = np.mean(predictions)
            #print(final_engagement_prediction)
            # Append LLM's prediction to the state for future prediction
            arm_state.append(50 if final_engagement_prediction > 0.5 else 0)  # Use the predicted state (engagement) as input for the next time step, 0.5 will considered as engaged
            #print(arm_state)
            arm_action.append(action_trajectories[arm][t + args.t1])  # Keep action sequence unchanged

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


def process_arm(arm, w, args, features, state_trajectories, action_trajectories, sys_prompt, config, prompt_templates, starting_prompt_templates, extraction_failures):
    """Process a single arm --> for parallelisation."""
    
    week_steps = np.arange(40)
    arm_features = features[arm]
    individual_predictions = []  # Store individual predictions for this arm and week
    predictions = []

    if w == 0:
        mapped_features = map_features_to_prompt(arm_features, [], [], first_week=True)
        prompt_text = [generate_prompt(mapped_features, prompt_template) for prompt_template in starting_prompt_templates]
    else:
        history_end_week = week_steps[w - 1]
        arm_states = state_trajectories[arm][:history_end_week]
        arm_actions = action_trajectories[arm][:history_end_week]
        mapped_features = map_features_to_prompt(arm_features, arm_states, arm_actions)
        prompt_text = [generate_prompt(mapped_features, prompt_template) for prompt_template in prompt_templates]

    # Compute ground truth for this arm
    ground_truth = 1 if np.array(state_trajectories[arm][w]) > 30 else 0

    # Collect predictions
    for prompt in prompt_text:
        responses = []
        for _ in range(args.num_queries):
            engagement_prediction, response = LLM_eval(config, args.config_path.split('_')[0], sys_prompt, prompt)
            if engagement_prediction == "error":
                extraction_failures += 1
                engagement_prediction = 0
            responses.append(engagement_prediction)
        predictions.append(np.mean(responses))  # Average responses for this prompt template
        individual_predictions.append(responses)  # Save individual predictions for each prompt

    final_engagement_prediction = np.mean(predictions)

    return arm, ground_truth, final_engagement_prediction, individual_predictions


def process_data_weekly_with_prompt_ensemble(args, config, features, state_trajectories, action_trajectories, 
                                              sys_prompt):
    # Use all prompt versions in the ensemble
    prompt_templates = [bin_prompt_v1(), bin_prompt_v2(), bin_prompt_v3(), bin_prompt_v4(), bin_prompt_v5()] 
    starting_prompt_templates = [starting_prompt_v2(), starting_prompt_v3(), starting_prompt_v4(), starting_prompt_v5(), starting_prompt_v6()]

    all_binary_predictions = [[] for _ in range(args.t2)]  # To store binary predictions for t2 steps
    all_ground_truths = [[] for _ in range(args.t2)]
    all_individual_predictions = []  # Store all individual predictions
    extraction_failures = 0
    structured_results = {}
    ground_truths = []

    model = args.config_path.split('_')[0]
    output_dir = f"./results/weekly/{model}_{args.num_arms}"
    os.makedirs(output_dir, exist_ok=True)

    # Check if already run --> resume from last completed week
    saved_weeks = glob.glob(f"{output_dir}/structured_results_t1_{args.t1}_t2_{args.t2}_week_*.json")
    if saved_weeks:
        # Get week numbers from saved file names
        completed_weeks = sorted([int(f.split("_week_")[1].split(".")[0]) for f in saved_weeks])
        last_completed_week = completed_weeks[-1]
        print(f"Resuming from week {last_completed_week + 1}")
    else: # no saved data
        last_completed_week = -1

    if last_completed_week != -1:
        # Load previously saved results
        with open(f"{output_dir}/binary_predictions_t1_{args.t1}_t2_{args.t2}_week_{last_completed_week}.json", "r") as f:
            all_binary_predictions = json.load(f)

        with open(f"{output_dir}/ground_truths_t1_{args.t1}_t2_{args.t2}_week_{last_completed_week}.json", "r") as f:
            all_ground_truths = json.load(f)

        with open(f"{output_dir}/structured_results_t1_{args.t1}_t2_{args.t2}_week_{last_completed_week}.json", "r") as f:
            structured_results = json.load(f)

    for w in tqdm(range(last_completed_week + 1, args.t2), desc="Processing weeks", leave=False, file=sys.stdout):

        if w < args.t1:  # Skip LLM predictions for months before t1
            continue

        if args.t1 <= w <= args.t2:
            structured_results[w] = {}

            # Parallelised loop for arms
            saved_arm_files = glob.glob(f"{output_dir}/arm_results_week_{w}_arm_*.json")
            completed_arms = set(int(f.split("_arm_")[1].split(".")[0]) for f in saved_arm_files)

            with ThreadPoolExecutor(max_workers=10) as executor:
                arms_progress = tqdm(total=args.num_arms, desc=f"Week {w} arms", leave=False, file=sys.stdout)
                futures = [
                    executor.submit(
                        process_arm,
                        arm,
                        w,
                        args,
                        features,
                        state_trajectories,
                        action_trajectories,
                        sys_prompt,
                        config,
                        prompt_templates,
                        starting_prompt_templates,
                        extraction_failures
                    )
                    for arm in range(args.num_arms) if arm not in completed_arms
                ]
                for future in futures:
                    arm, ground_truth, final_engagement_prediction, individual_predictions = future.result()

                    print(f"Processed arm {arm} for week {w}")
                    arms_progress.update(1)
                    
                    # Save intermediate arm results --> need to add a bash script for autodeletion of these after saving weekly!!
                    arm_result_path = f"{output_dir}/arm_results_week_{w}_arm_{arm}.json"
                    with open(arm_result_path, "w") as f:
                        json.dump({
                            "arm": arm,
                            "week": w,
                            "ground_truth": ground_truth,
                            "final_engagement_prediction": final_engagement_prediction,
                            "individual_predictions": individual_predictions
                        }, f, indent=4)

                    # Store results
                    structured_results[w][arm] = {"responses": individual_predictions}
                    all_binary_predictions[w].append(final_engagement_prediction)
                    all_ground_truths[w].append(ground_truth)
                    all_individual_predictions.append(individual_predictions)

                arms_progress.close()

            # Save intermediate results after each week
            with open(f"{output_dir}/structured_results_t1_{args.t1}_t2_{args.t2}_week_{w}.json", "w") as f:
                json.dump(structured_results, f, indent=4)

            with open(f"{output_dir}/binary_predictions_t1_{args.t1}_t2_{args.t2}_week_{w}.json", "w") as f:
                json.dump(all_binary_predictions, f, indent=4)

            with open(f"{output_dir}/ground_truths_t1_{args.t1}_t2_{args.t2}_week_{w}.json", "w") as f:
                json.dump(all_ground_truths, f, indent=4)

    # Save final results
    with open(f"{output_dir}/all_individual_predictions_t1_{args.t1}_t2_{args.t2}.json", "w") as json_file:
        json.dump(all_individual_predictions, json_file, indent=4)

    with open(f"{output_dir}/ground_truths_t1_{args.t1}_t2_{args.t2}.json", "w") as json_file:
        json.dump(ground_truths, json_file, indent=4)

    with open(f"{output_dir}/structured_results_t1_{args.t1}_t2_{args.t2}.json", "w") as f:
        json.dump(structured_results, f, indent=4)

    return (all_ground_truths, all_binary_predictions, extraction_failures)