# imports 
import argparse
import glob
import numpy as np

from aggregation import *
from plot import plot_performance_vs_month, plot_performance_vs_month_new, plot_distribution_over_time, plot_uncertainties_vs_month
from normalization import rank_normalization, z_score_normalization, log_normalization, min_max_normalization

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_arms", type=int, default=100, help="Number of mothers to simulate.")
    parser.add_argument("--models", nargs='+', default=["openai", "anthropic", "meta"], help="List of LLM models used in evals.")

    args = parser.parse_args()

    # Dictionary to store results for each model
    model_results = {
        model: {
            "predictions": sorted(glob.glob(f"./results/{model}_{args.num_arms}/all_individual_predictions_t1_*_t2_*.json")),
            "ground_truths": sorted(glob.glob(f"./results/{model}_{args.num_arms}/ground_truths_t1_*_t2_*.json")),
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

    # Track ground truths for consistency check and plotting
    gcs_over_time = []

    # Iterate over months (assumed consistent across all models)
    num_files = len(model_results[args.models[0]]["predictions"])
    months = np.arange(num_files)

    # Process each month (or time step)
    for month in range(num_files):
        # For aggregation and averaging
        mean_predictions_all_models = []
        epistemic_uncertainties_all_models = []

        # Loop over each model to load predictions and uncertainties
        for model in args.models:
            model_data = model_results[model]

            # Load predictions and ground truths
            all_individual_preds, ground_truths = load_predictions_and_ground_truths(
                model_data["predictions"][month], model_data["ground_truths"][month]
            )
            ground_truths = np.squeeze(ground_truths)

            # Store ground truths once
            if model == args.models[0]:
                gcs_over_time.append(ground_truths)

            # Mean predictions
            mean_predictions = np.array([np.mean(predictions) for predictions in all_individual_preds])

            # Calculate uncertainties
            epistemic_uncertainty, _, _ = compute_uncertainties_from_llm_predictions(all_individual_preds)
            epistemic_uncertainty = np.squeeze(epistemic_uncertainty)

            # Store metrics for each model
            acc, f1, log_likelihood = compute_metrics(mean_predictions, ground_truths)
            model_data["accuracies"].append(acc)
            model_data["f1_scores"].append(f1)
            model_data["log_likelihoods"].append(log_likelihood)
            model_data["epistemic_uncertainty"].append(epistemic_uncertainty)
            model_data["mean_predictions"].append(mean_predictions)

            # Add to lists for aggregation
            mean_predictions_all_models.append(mean_predictions)
            epistemic_uncertainties_all_models.append(epistemic_uncertainty)

        # Aggregate predictions using Bayesian aggregation and direct averaging
        # P_combined = bayesian_aggregation(*mean_predictions_all_models, *epistemic_uncertainties_all_models, normalization_method=rank_normalization)
        P_combined = bayesian_aggregation(predictions=mean_predictions_all_models,
                                          uncertainties=epistemic_uncertainties_all_models,
                                          normalization_method=rank_normalization
                                         )
        P_direct_avg = infer_posterior(*mean_predictions_all_models)

        # Calculate metrics for aggregated predictions
        acc_agg, f1_agg, log_likelihood_agg = compute_metrics(P_combined, ground_truths)
        acc_avg, f1_avg, log_likelihood_avg = compute_metrics(P_direct_avg, ground_truths)

        # Store aggregated metrics
        accuracies_agg.append(acc_agg)
        accuracies_avg.append(acc_avg)
        f1_scores_agg.append(f1_agg)
        f1_scores_avg.append(f1_avg)
        log_likelihoods_agg.append(log_likelihood_agg)
        log_likelihoods_avg.append(log_likelihood_avg)


    # Plot performance curves for each metric
    aggregated_metrics = {
        "Accuracy": accuracies_agg,
        "F1 Score": f1_scores_agg,
        "Log Likelihood": log_likelihoods_agg
    }
    averaged_metrics = {
        "Accuracy": accuracies_avg,
        "F1 Score": f1_scores_avg,
        "Log Likelihood": log_likelihoods_avg
    }
    for metric_name, metric_key in zip(["Accuracy", "F1 Score", "Log Likelihood"],
                                    ["accuracies", "f1_scores", "log_likelihoods"]):
        model_metrics = [model_results[model][metric_key] for model in args.models]
        plot_performance_vs_month_new(months, *model_metrics, 
                                    metric_agg=aggregated_metrics[metric_name], 
                                    metric_avg=averaged_metrics[metric_name], 
                                    metric_name=metric_name, 
                                    model_labels=args.models)


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