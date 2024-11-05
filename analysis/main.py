# imports 
import argparse
import glob

from aggregation import *
from plot import plot_performance_vs_month, plot_distribution_over_time, plot_uncertainties_vs_month
from normalization import rank_normalization, z_score_normalization, log_normalization, min_max_normalization

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_arms", type=int, default=100, help="Number of mothers to simulate.")
    parser.add_argument("--model", type=str, default="openai", help="LLM model used in evals.")
    parser.add_argument("--t1", type=int, default=0, help="Start month for LLM predictions.")
    parser.add_argument("--t2", type=int, default=4, help="End month for LLM predictions.")

    args = parser.parse_args()

    months = [2, 3, 4, 5, 6, 7, 8]

    # Specify results dir 
    results_dir = f'./results/{args.model}_{args.num_arms}'

    # Get all JSON files that match the patterns for predictions and ground truths
    prediction_files = sorted(glob.glob(f"{results_dir}/all_individual_predictions_t1_*_t2_*.json"))
    ground_truth_files = sorted(glob.glob(f"{results_dir}/ground_truths_t1_*_t2_*.json"))

    # Split into LLM-only files (first two months) and the rest
    prediction_llm_paths = prediction_files[:2]  # First two files for LLM-only
    ground_truths_llm_paths = ground_truth_files[:2] 

    # Remaining months for LLM + NN + Aggregated + Direct Averaging
    predictions_paths = prediction_files[2:]
    ground_truths_paths = ground_truth_files[2:]

    # NN, aggregated, Direct Averaging not available in the first two months
    accuracies_llm, accuracies_nn, accuracies_agg, accuracies_avg = [], [None, None], [None, None], [None, None]
    f1_scores_llm, f1_scores_nn, f1_scores_agg, f1_scores_avg = [], [None, None], [None, None], [None, None]
    log_likelihoods_llm, log_likelihoods_nn, log_likelihoods_agg, log_likelihoods_avg = [], [None, None], [None, None], [None, None]

    
    # Handle the first two months (LLM only)
    for prediction_llm_path, ground_truths_llm_path in zip(prediction_llm_paths, ground_truths_llm_paths):
        # Load LLM predictions and ground truths
        all_individual_predictions_llm, ground_truths_llm = load_predictions_and_ground_truths(prediction_llm_path, ground_truths_llm_path)
        ground_truths_llm = np.squeeze(ground_truths_llm)
        
        # Compute mean LLM predictions
        mean_predictions_llm = np.array([np.mean(predictions) for predictions in all_individual_predictions_llm])
        
        # Compute LLM performance metrics
        acc_llm, f1_llm, log_likelihood_llm = compute_metrics(mean_predictions_llm, ground_truths_llm)
        
        # Store the results
        accuracies_llm.append(acc_llm)
        f1_scores_llm.append(f1_llm)
        log_likelihoods_llm.append(log_likelihood_llm)

    
    # store percentage cases when lower uncertainty prediction was actually selected
    selected_confidence_llm_over_time = [] 
    selected_confidence_mc_over_time = []
    # store percentage cases when it would have been better to select other model's prediction in aggregation
    improved_selection_llm_over_time = [] 
    improved_selection_mc_over_time = []
    # store distributions
    epistemic_uncertainty_llm_over_time = []
    epistemic_uncertainty_nn_over_time = []
    mean_predictions_llm_over_time = []
    mean_predictions_nn_over_time = []
    # store groud truths
    gcs_over_time = []
    
    # Handle months 2 to 7 (LLM + NN + Aggregated + Direct Averaging)
    for month, predictions_path, ground_truths_path in zip(months, predictions_paths, ground_truths_paths):
        # Load MC-Dropout predictions and ground truths for the neural network (NN)
        mc_predictions, ground_truths_nn = load_mc_predictions_and_ground_truths(month)
        mean_predictions_nn = np.mean(mc_predictions, axis=0).squeeze()
        ground_truths_nn = np.squeeze(ground_truths_nn)

        # Compute uncertainties for the NN predictions
        epistemic_uncertainty_nn, aleatoric_uncertainty_nn, predictive_uncertainty_nn = compute_uncertainties(mc_predictions)
        epistemic_uncertainty_nn = np.squeeze(epistemic_uncertainty_nn)

        # Load LLM predictions and ground truths
        all_individual_predictions_llm, ground_truths_llm = load_predictions_and_ground_truths(predictions_path, ground_truths_path)
        ground_truths_llm = np.squeeze(ground_truths_llm)

        assert np.all(np.array(ground_truths_llm) == ground_truths_nn), "Ground truths are not the same!"

        # Compute mean LLM predictions
        mean_predictions_llm = np.array([np.mean(predictions) for predictions in all_individual_predictions_llm])

        # Compute uncertainties from LLM predictions
        epistemic_uncertainty_llm, aleatoric_uncertainty_llm, predictive_uncertainty_llm = compute_uncertainties_from_llm_predictions(all_individual_predictions_llm)
        epistemic_uncertainty_llm = np.squeeze(epistemic_uncertainty_llm)
        
        # Aggregate the predictions using Bayesian aggregation
        P_combined = bayesian_aggregation(mean_predictions_llm, mean_predictions_nn, epistemic_uncertainty_llm, epistemic_uncertainty_nn, normalization_method=rank_normalization)
        
        # Direct averaging of LLM and NN predictions
        P_direct_avg = infer_posterior(mean_predictions_llm, mean_predictions_nn)

        # Compute performance metrics for LLM, NN, aggregated, and direct averaging predictions
        acc_llm, f1_llm, log_likelihood_llm = compute_metrics(mean_predictions_llm, ground_truths_llm)
        acc_nn, f1_nn, log_likelihood_nn = compute_metrics(mean_predictions_nn, ground_truths_nn)
        acc_agg, f1_agg, log_likelihood_agg = compute_metrics(P_combined, ground_truths_nn)
        acc_avg, f1_avg, log_likelihood_avg = compute_metrics(P_direct_avg, ground_truths_nn)

        # Store distributions for epistemic uncertainty and mean predictions
        epistemic_uncertainty_llm_over_time.append(epistemic_uncertainty_llm)
        epistemic_uncertainty_nn_over_time.append(epistemic_uncertainty_nn)
        mean_predictions_llm_over_time.append(mean_predictions_llm)
        mean_predictions_nn_over_time.append(mean_predictions_nn)

        # Store the results for each method
        accuracies_llm.append(acc_llm)
        accuracies_nn.append(acc_nn)
        accuracies_agg.append(acc_agg)
        accuracies_avg.append(acc_avg)

        f1_scores_llm.append(f1_llm)
        f1_scores_nn.append(f1_nn)
        f1_scores_agg.append(f1_agg)
        f1_scores_avg.append(f1_avg)

        log_likelihoods_llm.append(log_likelihood_llm)
        log_likelihoods_nn.append(log_likelihood_nn)
        log_likelihoods_agg.append(log_likelihood_agg)
        log_likelihoods_avg.append(log_likelihood_avg)

        ## Uncertainty analysis: 
        # Find indices where one model is correct and the other is incorrect
        correct_llm_incorrect_mc, correct_mc_incorrect_llm = identify_discrepancies(mean_predictions_llm, mean_predictions_nn, ground_truths_nn)

        # Analyse how often the LOWER uncertainty prediction is selected and correct
        selected_llm_confidence = compare_confidence(correct_llm_incorrect_mc, mean_predictions_llm, mean_predictions_nn, epistemic_uncertainty_llm, epistemic_uncertainty_nn, ground_truths_nn, eval_by='uncertainty')
        selected_mc_confidence = compare_confidence(correct_mc_incorrect_llm, mean_predictions_llm, mean_predictions_nn, epistemic_uncertainty_llm, epistemic_uncertainty_nn, ground_truths_nn, eval_by='uncertainty')
        selected_confidence_llm_over_time.append(selected_llm_confidence)
        selected_confidence_mc_over_time.append(selected_mc_confidence)

        # How often the OTHER model's prediction would have been better
        improve_with_llm, improve_with_mc = analyze_improvement(correct_llm_incorrect_mc, correct_mc_incorrect_llm, P_combined, ground_truths_nn)
        improved_selection_llm_over_time.append(improve_with_llm)
        improved_selection_mc_over_time.append(improve_with_mc)

        # Store ground truths
        gcs_over_time.append(ground_truths_nn)


    # Plot population
    plot_distribution_over_time(gcs_over_time, months, 
                                "Population Growth over Time", "Month", ylabel='Population', type='hist')

    # Plot uncertainty distributions over time
    plot_distribution_over_time(epistemic_uncertainty_nn_over_time, months, 
                                "2-stage Model Epistemic Uncertainty Development", "Epistemic Uncertainty")
    # plot_distribution_over_time(mean_predictions_nn_over_time, months, 
    #                             "2-stage Model Average Probability Development", "Avg. Probability")
    # convert to confidence
    confidence_nn_over_time = [np.where(x > 0.5, x, 1 - x) for x in mean_predictions_nn_over_time]
    plot_distribution_over_time(confidence_nn_over_time, months, 
                                "2-stage Model Confidence Development", "Confidence")
    plot_distribution_over_time(epistemic_uncertainty_llm_over_time, months, 
                                "LLM Epistemic Uncertainty Development", "Epistemic Uncertainty")
    # plot_distribution_over_time(mean_predictions_llm_over_time, months, 
    #                             "LLM Average Probability Development", "Avg. Probability")
    confidence_llm_over_time = [np.where(x > 0.5, x, 1 - x) for x in mean_predictions_llm_over_time]
    plot_distribution_over_time(confidence_llm_over_time, months, 
                                "LLM Confidence Development", "Confidence")
    
    # Plot uncertainty analysis over time 
    plot_uncertainties_vs_month(months, selected_confidence_llm_over_time, selected_confidence_mc_over_time, 
                                "Correct Lower Uncertainty Selections",
                                "correct_selections")
    plot_uncertainties_vs_month(months, improved_selection_llm_over_time, improved_selection_mc_over_time, 
                                "Potential Aggregation Improvements",
                                "aggregation_improvements")

    # Extend months to include the first two months
    all_months = [0, 1] + months

    # Plot the performance curves for each metric
    plot_performance_vs_month(all_months, accuracies_llm, accuracies_nn, accuracies_agg, accuracies_avg, "Accuracy")
    plot_performance_vs_month(all_months, f1_scores_llm, f1_scores_nn, f1_scores_agg, f1_scores_avg, "F1 Score")
    plot_performance_vs_month(all_months, log_likelihoods_llm, log_likelihoods_nn, log_likelihoods_agg, log_likelihoods_avg, "Log Likelihood")