# imports 
import argparse
import numpy as np

from aggregation import *
from plot import plot_uncertainty_over_time, plot_performance_vs_month_new
from normalization import rank_normalization

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_arms", type=int, default=500, help="Number of mothers to simulate.")
    parser.add_argument("--models", nargs='+', default=["anthropic", "google", "googlepro", "openai", "openaiheavy"], help="List of LLM models used in evals.")
    parser.add_argument("--t1", type=int, default=0, help="Start month for LLM predictions.")
    parser.add_argument("--t2", type=int, default=40, help="End month for LLM predictions.")

    args = parser.parse_args()

    # Dictionary to store results for each model
    model_results = {
        model: {
            "predictions": [f"./results/weekly/{model}_{args.num_arms}/all_individual_predictions_t1_{args.t1}_t2_{args.t2}.json"],
            "ground_truths": [f"./results/weekly/{model}_{args.num_arms}/ground_truths_t1_{args.t1}_t2_{args.t2}_week_{args.t2-1}.json"],
            "accuracies": [],
            "f1_scores": [],
            "log_likelihoods": [],
            "epistemic_uncertainty": [],
            "mean_predictions": []
        } for model in args.models
    }

    # Storage for aggregated results
    accuracies_agg, accuracies_avg = [], []
    f1_scores_agg, f1_scores_avg = [], []
    log_likelihoods_agg, log_likelihoods_avg = [], []

    for model in args.models:
        model_data = model_results[model] # dict for this model

        # Load predictions and ground truths
        # all_individual_preds : all 25 predictions for each arm and timestep [[],[],[],[],[]]
        all_individual_preds, ground_truths = load_predictions_and_ground_truths(
            model_data["predictions"][0], model_data["ground_truths"][0]
        )

        # print("RESHAPED ------------")
        all_individual_preds = np.reshape(all_individual_preds, (args.t2-args.t1, args.num_arms, 5, 5))
        ground_truths = np.squeeze(ground_truths)
        # print(f"predictions: {len(all_individual_preds)}, {len(all_individual_preds[0])}, {len(all_individual_preds[0][0])}, {len(all_individual_preds[0][0][0])}")
        # print(f"Ground truths: {len(ground_truths)}, {len(ground_truths[0])}") 

        # Loop over timesteps and arms, use ground truths to compare with predictions
        final_preds_all_time = []
        for t in range(args.t2-args.t1):
            gcs_at_t = ground_truths[t]
            final_preds_all_arms = []

            for arm in range(args.num_arms):
                prompts_runs = all_individual_preds[t][arm] # 5 * 5
                mean_predictions_per_prompt = np.mean(prompts_runs, axis=1) 
                final_mean_prediction = np.mean(mean_predictions_per_prompt) 
                final_preds_all_arms.append(final_mean_prediction)

            #     print(f"Timestep {t}, Arm {arm}")
            #     print(f"Mean predictions per prompt: {mean_predictions_per_prompt}")
            #     print(f"Final mean prediction for this arm: {final_mean_prediction}")

            # print("-----------")
            # print(f"Final arms predictions: {final_preds_all_arms}")
            final_preds_all_time.append(final_preds_all_arms)

        # Calculate uncertainties
        timestep_ind_preds = restructure_predictions(all_individual_preds)
        for time_mat in timestep_ind_preds:
            epistemic_uncertainty, _, _ = compute_uncertainties_from_llm_predictions(time_mat) 
            epistemic_uncertainty = np.squeeze(epistemic_uncertainty)
            model_data["epistemic_uncertainty"].append(epistemic_uncertainty)

        # Loop over timesteps for accuracy and aggregation calcs
        for t in range(args.t2-args.t1):  
            preds_at_t = [final_preds_all_time[t][arm] for arm in range(args.num_arms)]  
            gcs_at_t = ground_truths[t, :]  

            # Calc epistemic uncertainty and metrics for this timestep
            acc, f1, log_likelihood = compute_metrics(preds_at_t, gcs_at_t)

            # Append metrics for plotting
            model_data["accuracies"].append(acc)
            model_data["f1_scores"].append(f1)
            model_data["log_likelihoods"].append(log_likelihood)
            model_data["mean_predictions"].append(preds_at_t)

    
    # print("Model results: ", model_results)

    results_for_aggregation = {}
    uncertainties_for_aggregation = {}
    for model in args.models:
        for i, (result, uncertainty) in enumerate(zip(model_results[model]["mean_predictions"], model_results[model]["epistemic_uncertainty"])):
            results_for_aggregation.setdefault(i, []).append(result)
            uncertainties_for_aggregation.setdefault(i, []).append(uncertainty)

    
    P_combined, unc_combined = [], []
    P_direct_avg, unc_direct_avg = [], []
    P_lowest_unc, unc_lowest_unc = [], []
    for t in range(args.t2-args.t1):
        combined, unc = bayesian_aggregation(predictions=results_for_aggregation[t],
                                             uncertainties=uncertainties_for_aggregation[t],
                                             normalization_method=rank_normalization
                                            )
        P_combined.append(combined), unc_combined.append(unc)
        
        # flattened_predictions = [np.array(p).flatten() for p in results_for_aggregation[t]] 
        # # flattened_predictions --> 500 predictions for each model at timestep t
        # # print(flattened_predictions, len(flattened_predictions[0]))
        # flattened_uncertainties = [np.array(u).flatten() for u in uncertainties_for_aggregation[t]]
        # lowest_unc, lowest_unc_unc = infer_posterior(*flattened_predictions, 
        #                                              uncertainties=flattened_uncertainties, 
        #                                              normalization_method=rank_normalization)
        
        direct_avg, direct_avg_unc = direct_averaging(predictions=results_for_aggregation[t],
                                                      uncertainties=uncertainties_for_aggregation[t],
                                                      normalization_method=rank_normalization
                                                     )
        P_direct_avg.append(direct_avg), unc_direct_avg.append(direct_avg_unc)

        lowest_unc, lowest_unc_unc = uncertainty_based_selection(predictions=results_for_aggregation[t],
                                                                 uncertainties=uncertainties_for_aggregation[t])
        P_lowest_unc.append(lowest_unc), unc_lowest_unc.append(lowest_unc_unc)

    # print("P_combined: ", len(P_combined), len(P_combined[0])) # 500 * 40
    # print("ground_truths: ", len(ground_truths), len(ground_truths[0])) # 40

    # Calculate metrics for aggregated predictions
    all_acc_agg, all_f1_agg, all_log_likelihood_agg = [], [], []
    all_acc_avg, all_f1_avg, all_log_likelihood_avg = [], [], []
    all_acc_low, all_f1_low, all_log_likelihood_low = [], [], []
    for i in range(len(P_combined)):
        acc_agg, f1_agg, log_likelihood_agg = compute_metrics(P_combined[i], ground_truths[i])
        all_acc_agg.append(acc_agg), all_f1_agg.append(f1_agg), all_log_likelihood_agg.append(log_likelihood_agg)

        acc_avg, f1_avg, log_likelihood_avg = compute_metrics(P_direct_avg[i], ground_truths[i])
        all_acc_avg.append(acc_avg), all_f1_avg.append(f1_avg), all_log_likelihood_avg.append(log_likelihood_avg)

        acc_lowest_unc, f1_lowest_unc, log_likelihood_lowest_unc = compute_metrics(P_lowest_unc[i], ground_truths[i])
        all_acc_low.append(acc_lowest_unc), all_f1_low.append(f1_lowest_unc), all_log_likelihood_low.append(log_likelihood_lowest_unc)

    # Plot performance curves for each metric
    timesteps = np.arange(args.t1, args.t2)
    aggregated_metrics = {
        "Accuracy": all_acc_agg,
        "F1 Score": all_f1_agg,
        "Log Likelihood": all_log_likelihood_agg
    }
    averaged_metrics = {
        "Accuracy": all_acc_avg,
        "F1 Score": all_f1_avg,
        "Log Likelihood": all_log_likelihood_avg
    }
    lowest_unc_metrics = {
        "Accuracy": all_acc_low,
        "F1 Score": all_f1_low,
        "Log Likelihood": all_log_likelihood_low
    }

    for metric_name, metric_key in zip(["Accuracy", "F1 Score", "Log Likelihood"],
                                    ["accuracies", "f1_scores", "log_likelihoods"]):
        model_metrics = [model_results[model][metric_key] for model in args.models]

        plot_performance_vs_month_new(
            timesteps, 
            *model_metrics,
            metric_agg=[np.mean(agg_metric) for agg_metric in aggregated_metrics[metric_name]],
            metric_avg=[np.mean(avg_metric) for avg_metric in averaged_metrics[metric_name]],
            metric_low=[np.mean(low_metric) for low_metric in lowest_unc_metrics[metric_name]],
            metric_name=metric_name,
            model_labels=args.models,
            separate_axes=True
        )

    # Plot uncertainty over time
    plot_uncertainty_over_time(
        timesteps=timesteps,
        model_results=model_results,
        combined_uncertainty=unc_combined,
        direct_avg_uncertainty=unc_direct_avg,
        lowest_uncertainty=unc_lowest_unc,
        models=args.models
    )



        
    # # Plot correct lowest-k selections for each model
    # model_predictions = {
    #     model: model_results[model]["mean_predictions"] for model in args.models
    # }
    # model_predictions["Posterior (Avg)"] = [P_direct_avg]
    # model_predictions["Uncertainty-weighted Posterior"] = [P_combined]

    # print("Model predictions for plot_k_corresponding:", model_predictions)
    # print("Ground truths for plot_k_corresponding:", gcs_over_time)

    # plot_k_corresponding(model_predictions, gcs_over_time, 5)


    # Plot uncertainty distributions over time for each model
    # for model in args.models:
    #     plot_distribution_over_time(model_results[model]["epistemic_uncertainty"], months,
    #                                 f"{model} Model Epistemic Uncertainty Development", "Epistemic Uncertainty")

    # # Plot population
    # plot_distribution_over_time(gcs_over_time, months, 
    #                             "Population Growth over Time", "Month", ylabel='Population', type='hist')

    # # Plot uncertainty distributions over time
    # plot_distribution_over_time(meta_epistemic_uncertainty_over_time, months, 
    #                             f"{args.models[1]} Model Epistemic Uncertainty Development", "Epistemic Uncertainty")
    # # convert to confidence
    # confidence_nn_over_time = [np.where(x > 0.5, x, 1 - x) for x in meta_mean_predictions_over_time]
    # plot_distribution_over_time(confidence_nn_over_time, months, 
    #                             f"{args.models[1]} Model Confidence Development", "Confidence")
    
    # plot_distribution_over_time(openai_epistemic_uncertainty_over_time, months, 
    #                             f"{args.models[0]} Model Uncertainty Development", "Epistemic Uncertainty")
    # confidence_llm_over_time = [np.where(x > 0.5, x, 1 - x) for x in openai_mean_predictions_over_time]
    # plot_distribution_over_time(confidence_llm_over_time, months, 
    #                             "LLM Confidence Development", "Confidence")
    
    # # Plot uncertainty analysis over time 
    # plot_uncertainties_vs_month(months, selected_confidence_openai_over_time, selected_confidence_meta_over_time, 
    #                             "Correct Lower Uncertainty Selections",
    #                             "correct_selections")
    # plot_uncertainties_vs_month(months, improved_selection_openai_over_time, improved_selection_meta_over_time, 
    #                             "Potential Aggregation Improvements",
    #                             "aggregation_improvements")