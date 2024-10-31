import argparse
import openai
import os

import numpy as np

from LLM_simulator import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/trajectories_continuous.json", help="Path to the data file.")
    parser.add_argument("--num_arms", type=int, default=100, help="Number of mothers to simulate.")
    parser.add_argument("--prompt_version", type=int, default=6, help="Version of the prompt template to use.")
    parser.add_argument("--filter", type=bool, default=False, help="Whether to filter highly engaged arms.")
    parser.add_argument("--use_features", type=bool, default=True, help="Whether to use features in the prompt.")
    parser.add_argument("--config_path", type=str, default="openai_config.json", help="Path to the LLM API configuration file.")
    parser.add_argument("--t1", type=int, default=10, help="Number of time steps to use as input.")
    parser.add_argument("--t2", type=int, default=5, help="Number of time steps to predict.")

    args = parser.parse_args()

    # create the directory : model_num_arms_t1_t2
    model = args.config_path.split('_')[0]
    os.makedirs(f"{model}_{args.num_arms}_{args.t1}_{args.t2}", exist_ok=True)

    engine = load_config(args.config_path)

    features, state_trajectories, action_trajectories = data_preprocessing(args.data_path)

    if args.filter: 
        filtered_state_trajectories = []
        filtered_action_trajectories = []
        filtered_features = []
        count = 0
        for i in range(len(state_trajectories)):
            num_engaged = np.mean(np.array(state_trajectories[i][args.t1:args.t1+args.t2]) > 30)
            if num_engaged <= 0.8 and num_engaged >= 0.3:
                filtered_state_trajectories.append(state_trajectories[i])
                filtered_action_trajectories.append(action_trajectories[i])
                filtered_features.append(features[i])
                count += 1
            if count == args.num_arms:
                break
        
        state_trajectories = filtered_state_trajectories
        action_trajectories = filtered_action_trajectories
        features = filtered_features

    # Select the appropriate prompt template (must be for binary prediction)
    if args.prompt_version == 6:
        prompt_template = prompt_template_v6()
    elif args.prompt_version == 7:
        prompt_template = prompt_template_v7()
    elif args.prompt_version == 8:
        prompt_template = prompt_template_v8()
    else:       
        raise ValueError(f"Invalid prompt version: {args.prompt_version}")
    
    if args.use_features:
        sys_prompt = system_prompt()
    else:
        sys_prompt = ''

    # Process the data and compute errors
    (all_ground_truths, all_binary_predictions,
     accuracy_per_step, log_likelihood_per_step, f1_score_per_step, total_accuracy, 
     total_log_likelihood, total_f1_score, extraction_failures) = process_data(
        args, engine, features, state_trajectories, action_trajectories, prompt_template, sys_prompt
    )
    print(all_binary_predictions)
    print(all_ground_truths)
    # Save all predictions, ground truths, and binary predictions as numpy arrays
    np.save(f"./autoregressive_{args.t1}_{args.t2}/engagement_gpt_ground_truths_{args.filter}_{args.prompt_version}_use_features_{str(args.use_features)}_{timestamp}.npy", all_ground_truths)
    np.save(f"./autoregressive_{args.t1}_{args.t2}/engagement_gpt_binary_predictions_{args.filter}_{args.prompt_version}_use_features_{str(args.use_features)}_{timestamp}.npy", all_binary_predictions)

    # Save results to JSON
    results = {
        "accuracy_per_step": accuracy_per_step,
        "log_likelihood_per_step": log_likelihood_per_step,
        "total_accuracy": total_accuracy,
        "total_log_likelihood": total_log_likelihood,
        "total_f1_score": total_f1_score,
        "extraction_failures": extraction_failures,
        "ground_truths": all_ground_truths,
        "binary_predictions": all_binary_predictions,
    }

    # with open(f"engagement_gpt_predictions_results_{args.filter}_{args.prompt_version}_{timestamp}.json", "w") as outfile:
    #     json.dump(results, outfile, indent=4)

    # Log final results in a text file, also save the metrics per step
    with open(f"./autoregressive_{args.t1}_{args.t2}/engagement_gpt_predictions_summary_{args.prompt_version}_filter_{str(args.filter)}_{timestamp}.txt", "w") as summary_file:
        summary_file.write(f"Total Accuracy: {total_accuracy}\n")
        summary_file.write(f"Total Log Likelihood: {total_log_likelihood}\n")
        summary_file.write(f"Total F1 Score: {total_f1_score}\n")
        summary_file.write("Metrics per step:\n")  
        for i in range(len(accuracy_per_step)):
            summary_file.write(f"Step {i}:\n")
            summary_file.write(f"Accuracy: {accuracy_per_step[i]}\n")
            summary_file.write(f"Log Likelihood: {log_likelihood_per_step[i]}\n")
            summary_file.write(f"F1 Score: {f1_score_per_step[i]}\n")
            summary_file.write("\n")
        
        summary_file.write(f"Number of Extraction Failures: {extraction_failures}\n")

    print(f"Total Accuracy: {total_accuracy}")
    print(f"Total Log Likelihood: {total_log_likelihood}")
    print(f"Total F1 Score: {total_f1_score}")
    print(f"Number of Extraction Failures: {extraction_failures}")