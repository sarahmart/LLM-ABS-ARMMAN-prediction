import argparse
from LLM_simulator import *


def process_arm_with_action(arm, w, args, features, state_trajectories, action_trajectories, 
                            sys_prompt, config, prompt_templates, starting_prompt_templates, 
                            action_prompt_templates, include_actions_prompt_templates, 
                            starting_action_prompt_templates, extraction_failures):
    """Process a single arm --> for parallelisation, now incorporating actions."""
    
    week_steps = np.arange(40)
    arm_features = features[arm]
    individual_predictions = []  # Store individual predictions for this arm and week
    predictions = []

    # Determine if an action occurs at this timestep, or has occured previously for this arm
    action_taken = action_trajectories[arm][w] == 1 if w < len(action_trajectories[arm]) else False
    previous_action = action_trajectories[arm][w - 1] == 1 if w > 0 else False

    # Select appropriate prompt template
    if w == 0 and action_taken: # action in first week
        prompt_text = [generate_prompt(map_features_to_prompt(arm_features, [], [], first_week=True), 
                                       prompt_template) for prompt_template in starting_action_prompt_templates]
    elif w == 0: # no action, normal first week
        prompt_text = [generate_prompt(map_features_to_prompt(arm_features, [], [], first_week=True), 
                                       prompt_template) for prompt_template in starting_prompt_templates]
    elif action_taken: # action in this week
        history_end_week = week_steps[w - 1]
        arm_states = state_trajectories[arm][:history_end_week]
        arm_actions = action_trajectories[arm][:history_end_week]
        mapped_features = map_features_to_prompt(arm_features, arm_states, arm_actions)
        prompt_text = [generate_prompt(mapped_features, prompt_template) for prompt_template in action_prompt_templates]
    elif previous_action: # action in previous weeks
        history_end_week = week_steps[w - 1]
        arm_states = state_trajectories[arm][:history_end_week]
        arm_actions = action_trajectories[arm][:history_end_week]
        mapped_features = map_features_to_prompt(arm_features, arm_states, arm_actions)
        prompt_text = [generate_prompt(mapped_features, prompt_template) for prompt_template in include_actions_prompt_templates]
    else: # no action so far
        history_end_week = week_steps[w - 1]
        arm_states = state_trajectories[arm][:history_end_week]
        arm_actions = action_trajectories[arm][:history_end_week]
        mapped_features = map_features_to_prompt(arm_features, arm_states, arm_actions)
        prompt_text = [generate_prompt(mapped_features, prompt_template) for prompt_template in prompt_templates]

    # Compute ground truth for this arm
    ground_truth = 1 if np.array(state_trajectories[arm][w]) > 30 else 0

    # Collect predictions
    for prompt in prompt_text:
        print(prompt)
        responses = []
        for _ in range(args.num_queries):
            engagement_prediction, response = LLM_eval(config, args.config_path.split('_')[0], sys_prompt, prompt)
            print(response)
            if engagement_prediction == "error":
                extraction_failures += 1
                engagement_prediction = 0
            responses.append(engagement_prediction)
        predictions.append(np.mean(responses))  # Average responses for this prompt template
        individual_predictions.append(responses)  # Save individual predictions for each prompt

    final_engagement_prediction = np.mean(predictions)

    return arm, ground_truth, final_engagement_prediction, individual_predictions


def process_data_weekly_with_actions(args, config, 
                                     features, state_trajectories, action_trajectories, 
                                     sys_prompt):
    
    # Use all prompt versions in the ensemble
    prompt_templates = [bin_prompt_v1(), bin_prompt_v2(), bin_prompt_v3(), bin_prompt_v4(), bin_prompt_v5()] 
    starting_prompt_templates = [starting_prompt_v2(), starting_prompt_v3(), starting_prompt_v4(), starting_prompt_v5(), starting_prompt_v6()]
    action_prompt_templates = [action_prompt_v1(), action_prompt_v2(), action_prompt_v3(), action_prompt_v4(), action_prompt_v5()]
    include_actions_prompt_templates = [include_actions_prompt_v1(), include_actions_prompt_v2(), include_actions_prompt_v3(), include_actions_prompt_v4(), include_actions_prompt_v5()]
    starting_action_prompt_templates = [starting_action_prompt_v1(), starting_action_prompt_v2(), starting_action_prompt_v3(), starting_action_prompt_v4(), starting_action_prompt_v5()]

    all_binary_predictions = [[] for _ in range(args.t2)] 
    all_ground_truths = [[] for _ in range(args.t2)]
    all_individual_predictions = [] 
    extraction_failures = 0
    structured_results = {}
    ground_truths = []

    model = args.config_path.split('_')[0]
    output_dir = f"./results/actions/{model}_{args.num_arms}"
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
                        process_arm_with_action,
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
                        action_prompt_templates,  
                        include_actions_prompt_templates,
                        starting_action_prompt_templates,
                        extraction_failures
                    )
                    for arm in range(args.num_arms) if arm not in completed_arms
                ]
                for future in futures:
                    arm, ground_truth, final_engagement_prediction, individual_predictions = future.result()
                    print(arm, ground_truth, final_engagement_prediction, individual_predictions)

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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/trajectories_continuous.json", help="Path to the data file.")
    parser.add_argument("--num_arms", type=int, default=50, help="Number of mothers to simulate.")
    parser.add_argument("--config_path", type=str, default="openai_config.json", help="Path to the LLM API configuration file.")
    parser.add_argument("--t1", type=int, default=0, help="Start month for LLM predictions.")
    parser.add_argument("--t2", type=int, default=10, help="End month for LLM predictions.")
    parser.add_argument("--num_queries", type=int, default=5, help="Number of queries to LLM for each prompt.") 

    args = parser.parse_args()

    engine = load_config(args.config_path)

    features, state_trajectories, action_trajectories = data_preprocessing(args.data_path)

    sys_prompt = system_prompt()

    # Process the data and compute errors for monthly engagement with new mothers joining each month
    (all_ground_truths, 
     all_binary_predictions, 
     extraction_failures) = process_data_weekly_with_actions(args, 
                                                             engine, 
                                                             features, 
                                                             state_trajectories, 
                                                             action_trajectories, 
                                                             sys_prompt)
    
    # Compute total accuracy, F1 score, and log likelihood for months t1 to t2
    total_accuracy, total_f1_score, total_log_likelihood = compute_total_metrics(all_binary_predictions, all_ground_truths, args.t1, args.t2)

    # Flatten the lists before saving
    flat_ground_truths = [item for sublist in all_ground_truths for item in sublist]
    flat_binary_predictions = [item for sublist in all_binary_predictions for item in sublist]

    # Save to results/model_num_arms_t1_t2
    model = args.config_path.split('_')[0]
    # os.makedirs(f"results/{model}_{args.num_arms}_{args.t1}_{args.t2}", exist_ok=True)
    np.save(f"./results/actions/{model}_{args.num_arms}/engagement_gpt_ground_truths_t1_{args.t1}_t2_{args.t2}.npy", np.array(flat_ground_truths))
    np.save(f"./results/actions/{model}_{args.num_arms}/engagement_gpt_binary_predictions_t1_{args.t1}_t2_{args.t2}.npy", np.array(flat_binary_predictions))


    # Log final results in a text file, also save the metrics per step
    with open(f"./results/actions/{model}_{args.num_arms}/engagement_gpt_predictions_summary_t1_{args.t1}_t2_{args.t2}.txt", "w") as summary_file:
        summary_file.write(f"Total Accuracy: {total_accuracy}\n")
        summary_file.write(f"Total F1 Score: {total_f1_score}\n")
        summary_file.write(f"Total Log Likelihood: {total_log_likelihood}\n")
        summary_file.write(f"Number of Extraction Failures: {extraction_failures}\n")

    print(f"Total Accuracy: {total_accuracy}")
    print(f"Total F1 Score: {total_f1_score}")
    print(f"Total Log Likelihood: {total_log_likelihood}")
    print(f"Number of Extraction Failures: {extraction_failures}")