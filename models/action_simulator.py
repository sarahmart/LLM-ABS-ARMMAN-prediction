from LLM_simulator import *


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