import argparse
# import openai
# import os

import numpy as np

from LLM_simulator import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/trajectories_continuous.json", help="Path to the data file.")
    parser.add_argument("--num_arms", type=int, default=100, help="Number of mothers to simulate.")
    # parser.add_argument("--filter", type=bool, default=False, help="Whether to filter highly engaged arms.")
    # parser.add_argument("--use_features", type=bool, default=True, help="Whether to use features in the prompt.")
    parser.add_argument("--config_path", type=str, default="openai_config.json", help="Path to the LLM API configuration file.")
    parser.add_argument("--t1", type=int, default=0, help="Start month for LLM predictions.")
    parser.add_argument("--t2", type=int, default=4, help="End month for LLM predictions.")

    args = parser.parse_args()

    engine = load_config(args.config_path)

    features, state_trajectories, action_trajectories = data_preprocessing(args.data_path)

    # if args.filter: 
    #     filtered_state_trajectories = []
    #     filtered_action_trajectories = []
    #     filtered_features = []
    #     count = 0
    #     for i in range(len(state_trajectories)):
    #         num_engaged = np.mean(np.array(state_trajectories[i][args.t1:args.t1+args.t2]) > 30)
    #         if num_engaged <= 0.8 and num_engaged >= 0.3:
    #             filtered_state_trajectories.append(state_trajectories[i])
    #             filtered_action_trajectories.append(action_trajectories[i])
    #             filtered_features.append(features[i])
    #             count += 1
    #         if count == args.num_arms:
    #             break
        
        # state_trajectories = filtered_state_trajectories
        # action_trajectories = filtered_action_trajectories
        # features = filtered_features

    # # Select the appropriate prompt template (must be for binary prediction)
    # if args.prompt_version == 6:
    #     prompt_template = prompt_template_v6()
    # elif args.prompt_version == 7:
    #     prompt_template = prompt_template_v7()
    # elif args.prompt_version == 8:
    #     prompt_template = prompt_template_v8()
    # else:       
    #     raise ValueError(f"Invalid prompt version: {args.prompt_version}")
    
    # if args.use_features:
    #     sys_prompt = system_prompt()
    # else:
    #     sys_prompt = ''

    sys_prompt = system_prompt()

    # Process the data and compute errors for monthly engagement with new mothers joining each month
    (all_ground_truths, all_binary_predictions, extraction_failures) = process_data_monthly_with_prompt_ensemble(args, 
                                                                                                                 engine, 
                                                                                                                 features, 
                                                                                                                 state_trajectories, 
                                                                                                                 action_trajectories, 
                                                                                                                 sys_prompt,
                                                                                                                 initial_mothers=args.num_arms,
                                                                                                                 L=100, k=0.4, t0=10) # CHANGE LATER

    # Compute total accuracy, F1 score, and log likelihood for months t1 to t2
    total_accuracy, total_f1_score, total_log_likelihood = compute_total_metrics(all_binary_predictions, all_ground_truths, args.t1, args.t2)

    # Flatten the lists before saving
    flat_ground_truths = [item for sublist in all_ground_truths for item in sublist]
    flat_binary_predictions = [item for sublist in all_binary_predictions for item in sublist]

    # Save to results/model_num_arms_t1_t2
    model = args.config_path.split('_')[0]
    # os.makedirs(f"results/{model}_{args.num_arms}_{args.t1}_{args.t2}", exist_ok=True)
    np.save(f"./results/{model}_{args.num_arms}/engagement_gpt_ground_truths_t1_{args.t1}_t2_{args.t2}_{timestamp}.npy", np.array(flat_ground_truths))
    np.save(f"./results/{model}_{args.num_arms}/engagement_gpt_binary_predictions_t1_{args.t1}_t2_{args.t2}_{timestamp}.npy", np.array(flat_binary_predictions))


    # Log final results in a text file, also save the metrics per step
    with open(f"./results/{model}_{args.num_arms}/engagement_gpt_predictions_summary_t1_{args.t1}_t2_{args.t2}_{timestamp}.txt", "w") as summary_file:
        summary_file.write(f"Total Accuracy: {total_accuracy}\n")
        summary_file.write(f"Total F1 Score: {total_f1_score}\n")
        summary_file.write(f"Total Log Likelihood: {total_log_likelihood}\n")
        summary_file.write(f"Number of Extraction Failures: {extraction_failures}\n")

    print(f"Total Accuracy: {total_accuracy}")
    print(f"Total F1 Score: {total_f1_score}")
    print(f"Total Log Likelihood: {total_log_likelihood}")
    print(f"Number of Extraction Failures: {extraction_failures}")