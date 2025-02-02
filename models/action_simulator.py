import argparse
import pandas as pd

from atomicwrites import atomic_write

from LLM_simulator import *
from preprocess import * 


def process_arm_with_action(arm, w, args, features, state_trajectories, action_trajectories, 
                            sys_prompt, config, prompt_templates, starting_prompt_templates, 
                            action_prompt_templates, starting_action_prompt_templates, extraction_failures):
    """Process a single arm --> for parallelisation, now incorporating actions."""
    
    week_steps = np.arange(args.t2 - args.t1)
    arm_features = features[arm]
    individual_predictions = []  # Store individual predictions for this arm and week
    predictions = []

    # Determine if an action occurs at this timestep
    action_taken = w < len(action_trajectories[arm]) and action_trajectories[arm][w] == 1

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
    else: # no action in this step and not starting week 
        history_end_week = week_steps[w - 1]
        arm_states = state_trajectories[arm][:history_end_week]
        arm_actions = action_trajectories[arm][:history_end_week]
        mapped_features = map_features_to_prompt(arm_features, arm_states, arm_actions)
        prompt_text = [generate_prompt(mapped_features, prompt_template) for prompt_template in prompt_templates]

    # ground truth for this arm
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
        predictions.append(np.mean(responses))    # avg response for this prompt
        individual_predictions.append(responses)  # individual predictions for each prompt

    final_engagement_prediction = np.mean(predictions)

    return arm, ground_truth, final_engagement_prediction, individual_predictions


def process_data_weekly_with_actions(args, config, features, state_trajectories, action_trajectories, sys_prompt):
    prompt_templates = [bin_prompt_v1(), bin_prompt_v2(), bin_prompt_v3(), bin_prompt_v4(), bin_prompt_v5()]
    starting_prompt_templates = [starting_prompt_v2(), starting_prompt_v3(), starting_prompt_v4(), starting_prompt_v5(), starting_prompt_v6()]
    action_prompt_templates = [action_prompt_v1(), action_prompt_v2(), action_prompt_v3(), action_prompt_v4(), action_prompt_v5()]
    starting_action_prompt_templates = [starting_action_prompt_v1(), starting_action_prompt_v2(), starting_action_prompt_v3(), starting_action_prompt_v4(), starting_action_prompt_v5()]

    running_arms = len(state_trajectories)
    all_binary_predictions = [[] for _ in range(args.t2 - args.t1)]
    all_ground_truths = [[] for _ in range(args.t2 - args.t1)]
    all_individual_predictions = []
    extraction_failures = 0
    structured_results = {}
    ground_truths = []

    model = args.config_path.split('_')[0]
    output_dir = f"./results/actions/{model}_{args.num_arms}"
    os.makedirs(output_dir, exist_ok=True)

    def load_last_valid_state(output_dir, w):
        try:
            with open(f"{output_dir}/structured_results_t1_{args.t1}_t2_{args.t2}_week_{w-1}.json") as f:
                return json.load(f)
        except:
            return {}

    def save_arm_results_batch(results_buffer, output_dir, week):
        for result in results_buffer:
            arm = result['arm']
            with atomic_write(f"{output_dir}/arm_results_week_{week}_arm_{arm}.json", overwrite=True) as f:
                json.dump(result, f, indent=4)

    def cleanup_intermediate_files(output_dir, current_week):
        for f in glob.glob(f"{output_dir}/arm_results_week_{current_week-1}_arm_*.json"):
            os.remove(f)

    saved_weeks = glob.glob(f"{output_dir}/structured_results_t1_{args.t1}_t2_{args.t2}_week_*.json")
    if saved_weeks:
        completed_weeks = sorted([int(f.split("_week_")[1].split(".")[0]) for f in saved_weeks])
        last_completed_week = completed_weeks[-1]
        structured_results = load_last_valid_state(output_dir, last_completed_week + 1)
        print(f"Resuming from week {last_completed_week + 1}")

        with open(f"{output_dir}/binary_predictions_t1_{args.t1}_t2_{args.t2}_week_{last_completed_week}.json", "r") as f:
            all_binary_predictions = json.load(f)

        with open(f"{output_dir}/ground_truths_t1_{args.t1}_t2_{args.t2}_week_{last_completed_week}.json", "r") as f:
            all_ground_truths = json.load(f)

        failures_file = f"{output_dir}/extraction_failures_checkpoint.json"
        if os.path.exists(failures_file):
            with open(failures_file) as f:
                extraction_failures = json.load(f)['extraction_failures']
    else:
        last_completed_week = -1

    for w in tqdm(range(last_completed_week + 1, args.t2), desc="Processing weeks", leave=False, file=sys.stdout):
        if w < args.t1:
            continue

        if args.t1 <= w <= args.t2:
            structured_results[w] = {}
            saved_arm_files = glob.glob(f"{output_dir}/arm_results_week_{w}_arm_*.json")
            completed_arms = set(int(f.split("_arm_")[1].split(".")[0]) for f in saved_arm_files)
            arm_results_buffer = []

            with ThreadPoolExecutor(max_workers=10) as executor:
                arms_progress = tqdm(total=running_arms, desc=f"Week {w} arms", leave=False, file=sys.stdout)
                futures = [
                    executor.submit(
                        process_arm_with_action,
                        arm, w, args, features, state_trajectories, action_trajectories,
                        sys_prompt, config, prompt_templates, starting_prompt_templates,
                        action_prompt_templates, starting_action_prompt_templates, extraction_failures
                    )
                    for arm in range(running_arms) if arm not in completed_arms
                ]

                for future in futures:
                    arm, ground_truth, final_engagement_prediction, individual_predictions = future.result()
                    print(f"Processed arm {arm} for week {w}")
                    arms_progress.update(1)

                    result_dict = {
                        "arm": arm,
                        "week": w,
                        "ground_truth": ground_truth,
                        "final_engagement_prediction": final_engagement_prediction,
                        "individual_predictions": individual_predictions
                    }
                    arm_results_buffer.append(result_dict)

                    if len(arm_results_buffer) >= 10:
                        save_arm_results_batch(arm_results_buffer, output_dir, w)
                        arm_results_buffer.clear()

                    structured_results[w][arm] = {"responses": individual_predictions}
                    all_binary_predictions[w].append(final_engagement_prediction)
                    all_ground_truths[w].append(ground_truth)
                    all_individual_predictions.append(individual_predictions)

                    with atomic_write(f"{output_dir}/extraction_failures_checkpoint.json", overwrite=True) as f:
                        json.dump({"extraction_failures": extraction_failures}, f)

                arms_progress.close()

            # Save remaining results in buffer
            if arm_results_buffer:
                save_arm_results_batch(arm_results_buffer, output_dir, w)

            with atomic_write(f"{output_dir}/structured_results_t1_{args.t1}_t2_{args.t2}_week_{w}.json", overwrite=True) as f:
                json.dump(structured_results, f, indent=4)

            with atomic_write(f"{output_dir}/binary_predictions_t1_{args.t1}_t2_{args.t2}_week_{w}.json", overwrite=True) as f:
                json.dump(all_binary_predictions, f, indent=4)

            with atomic_write(f"{output_dir}/ground_truths_t1_{args.t1}_t2_{args.t2}_week_{w}.json", overwrite=True) as f:
                json.dump(all_ground_truths, f, indent=4)

            cleanup_intermediate_files(output_dir, w)

    # Save final results
    with atomic_write(f"{output_dir}/all_individual_predictions_t1_{args.t1}_t2_{args.t2}.json", overwrite=True) as f:
        json.dump(all_individual_predictions, f, indent=4)

    with atomic_write(f"{output_dir}/ground_truths_t1_{args.t1}_t2_{args.t2}.json", overwrite=True) as f:
        json.dump(ground_truths, f, indent=4)

    with atomic_write(f"{output_dir}/structured_results_t1_{args.t1}_t2_{args.t2}.json", overwrite=True) as f:
        json.dump(structured_results, f, indent=4)

    return (all_ground_truths, all_binary_predictions, extraction_failures)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/trajectories_continuous.json", help="Path to the data file.")
    parser.add_argument("--num_full_sample", type=int, default=500, help="Total number of mothers (used for demographic sample).")
    parser.add_argument("--num_arms", type=int, default=100, help="Number of mothers on whom to simulate actions.")
    parser.add_argument("--config_path", type=str, default="openai_config.json", help="Path to the LLM API configuration file.")
    parser.add_argument("--t1", type=int, default=0, help="Start month for LLM predictions.")
    parser.add_argument("--t2", type=int, default=15, help="End month for LLM predictions.")
    parser.add_argument("--num_queries", type=int, default=5, help="Number of queries to LLM for each prompt.") 

    args = parser.parse_args()

    engine = load_config(args.config_path)

    # get features, state_trajectories, action_trajectories from data for large sample
    features, state_trajectories, action_trajectories = data_preprocessing(args.data_path)
    features = features[:args.num_full_sample]
    state_trajectories = state_trajectories[0:args.num_full_sample]
    state_trajectories = [[1 if time > 30 else 0 for time in arm] for arm in state_trajectories]
    action_trajectories = action_trajectories[0:args.num_full_sample]

    # select all mothers who were actually acted on
    act_on = [i for i, arm in enumerate(action_trajectories) if any(arm)]
    # subset of actions for the mothers who were intervened on
    acted_actions = [action_trajectories[i] for i in act_on]

    # percentage of mothers acted on in each time step (all actions in first 6 time steps)
    counts = [sum(timestep) for timestep in zip(*acted_actions)]
    k = np.mean(counts[:6])/args.num_full_sample

    # get representative subset of num_arms (num acted on) mothers: k-means clustering with num_arms clusters
    selected_indices = select_representative_mothers(
        features_array=np.array(features),
        state_trajectories=state_trajectories,
        n_clusters=args.num_arms,
        random_state=42
    )

    representative_features = [features[i] for i in selected_indices]
    representative_states = [state_trajectories[i] for i in selected_indices]
    representative_actions = [action_trajectories[i] for i in selected_indices]
    
    # select a random subset of k*num_arms mothers to act on in each timestep 0-6: 
    num_to_act = int(np.round(k * args.num_arms))
    print("Number of mothers to act on in each timestep: ", num_to_act)

    selected_action_trajectories = np.zeros((len(representative_states), len(state_trajectories[0])))
    # mothers who will receive interventions in the new sim
    intervention_indices = [] 
    for t in range(6):
        available_indices = np.setdiff1d(np.arange(len(representative_states)), intervention_indices)
        rng = np.random.default_rng(42 + t)
        selected_for_action = rng.choice(available_indices, num_to_act, replace=False)
        
        print(f"Time {t}, selected mothers:", selected_for_action)
        
        intervention_indices.extend(selected_for_action)
        
        # Set action to 1 for selected mothers at this timestep
        for i in selected_for_action:
            selected_action_trajectories[i][t] = 1

    torun_actions = [selected_action_trajectories[i] for i in intervention_indices]
    torun_states = [representative_states[i] for i in intervention_indices]
    torun_features = [representative_features[i] for i in intervention_indices]
    
    # save actions to csv 
    stacked_actions = np.array(torun_actions)
    df = pd.DataFrame(stacked_actions)
    df.insert(0, "Mother Index", intervention_indices)  
    df.to_csv(f"./results/actions/action_trajectories_{len(torun_actions)}_t1_{args.t1}_t2_{args.t2}.csv", index=False)

    sys_prompt = system_prompt_action()

    # Process data
    (all_ground_truths, 
     all_binary_predictions, 
     extraction_failures) = process_data_weekly_with_actions(args, 
                                                             engine, 
                                                             torun_features, 
                                                             torun_states, 
                                                             torun_actions, 
                                                             sys_prompt)
    
    # Flatten before saving
    flat_ground_truths = [item for sublist in all_ground_truths for item in sublist]
    flat_binary_predictions = [item for sublist in all_binary_predictions for item in sublist]

    # Save to results/actions/model_num_arms_t1_t2
    model = args.config_path.split('_')[0]
    np.save(f"./results/actions/{model}_{args.num_arms}/engagement_gpt_ground_truths_t1_{args.t1}_t2_{args.t2}.npy", np.array(flat_ground_truths))
    np.save(f"./results/actions/{model}_{args.num_arms}/engagement_gpt_binary_predictions_t1_{args.t1}_t2_{args.t2}.npy", np.array(flat_binary_predictions))