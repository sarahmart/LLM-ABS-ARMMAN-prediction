# imports 
import argparse
import glob

from aggregation import *
from plot import plot_performance_vs_month, plot_distribution_over_time, plot_uncertainties_vs_month
from normalization import rank_normalization, z_score_normalization, log_normalization, min_max_normalization

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_arms", type=int, default=100, help="Number of mothers to simulate.")
    parser.add_argument("--models", nargs='+', default=["openai", "anthropic"], help="List of LLM models used in evals.")

    args = parser.parse_args()

    # OPENAI
    openai_dir = f'./results/{args.models[0]}_{args.num_arms}'
    print(openai_dir)
    prediction_files_openai = sorted(glob.glob(f"{openai_dir}/all_individual_predictions_t1_*_t2_*.json"))
    ground_truth_files_openai = sorted(glob.glob(f"{openai_dir}/ground_truths_t1_*_t2_*.json"))

    # META
    meta_dir = f'./results/{args.models[1]}_{args.num_arms}'
    print(meta_dir)
    prediction_files_meta = sorted(glob.glob(f"{meta_dir}/all_individual_predictions_t1_*_t2_*.json"))
    ground_truth_files_meta = sorted(glob.glob(f"{meta_dir}/ground_truths_t1_*_t2_*.json"))

    accuracies_openai, accuracies_meta, accuracies_agg, accuracies_avg = [], [], [], []
    f1_scores_openai, f1_scores_meta, f1_scores_agg, f1_scores_avg = [], [], [], []
    log_likelihoods_openai, log_likelihoods_meta, log_likelihoods_agg, log_likelihoods_avg = [], [], [], []
    
    # store percentage cases when lower uncertainty prediction was actually selected
    selected_confidence_openai_over_time, selected_confidence_meta_over_time = [], []
    # store percentage cases when it would have been better to select other model's prediction in aggregation
    improved_selection_openai_over_time, improved_selection_meta_over_time = [], []

    # store distributions
    openai_epistemic_uncertainty_over_time, meta_epistemic_uncertainty_over_time = [], []
    openai_mean_predictions_over_time, meta_mean_predictions_over_time = [], []
    # store ground truths
    gcs_over_time = []
    
    for openai_predictions_path, openai_ground_truths_path, meta_predictions_path, meta_ground_truths_path in zip(prediction_files_openai, 
                                                                                                                  ground_truth_files_openai,
                                                                                                                  prediction_files_meta,
                                                                                                                  ground_truth_files_meta):

        # Load openai predictions and ground truths
        openai_all_ind_preds, openai_ground_truths = load_predictions_and_ground_truths(openai_predictions_path, openai_ground_truths_path)
        openai_ground_truths = np.squeeze(openai_ground_truths)

        # Load meta predictions and ground truths
        meta_all_ind_preds, meta_ground_truths = load_predictions_and_ground_truths(meta_predictions_path, meta_ground_truths_path)
        meta_ground_truths = np.squeeze(meta_ground_truths)

        assert np.all(np.array(openai_ground_truths) == meta_ground_truths), "Ground truths are not the same!"

        # mean predictions
        openai_mean_predictions = np.array([np.mean(predictions) for predictions in openai_all_ind_preds])
        meta_mean_predictions = np.array([np.mean(predictions) for predictions in meta_all_ind_preds])

        # uncertainties from LLM predictions
        openai_epistemic_uncertainty, openai_aleatoric_uncertainty, openai_predictive_uncertainty = compute_uncertainties_from_llm_predictions(openai_all_ind_preds)
        openai_epistemic_uncertainty = np.squeeze(openai_epistemic_uncertainty)
        meta_epistemic_uncertainty, meta_aleatoric_uncertainty, meta_predictive_uncertainty = compute_uncertainties_from_llm_predictions(meta_all_ind_preds)
        meta_epistemic_uncertainty = np.squeeze(meta_epistemic_uncertainty)
        
        # Bayesian aggregation
        P_combined = bayesian_aggregation(openai_mean_predictions, meta_mean_predictions, openai_epistemic_uncertainty, meta_epistemic_uncertainty, normalization_method=rank_normalization)
        
        # Direct averaging 
        P_direct_avg = infer_posterior(openai_mean_predictions, meta_mean_predictions)

        # Compute performance metrics for LLMs, aggregated, and direct averaging predictions
        acc_openai, f1_openai, log_likelihood_openai = compute_metrics(openai_mean_predictions, openai_ground_truths)
        acc_meta, f1_meta, log_likelihood_meta = compute_metrics(meta_mean_predictions, meta_ground_truths)
        acc_agg, f1_agg, log_likelihood_agg = compute_metrics(P_combined, openai_ground_truths)
        acc_avg, f1_avg, log_likelihood_avg = compute_metrics(P_direct_avg, openai_ground_truths)

        # Store distributions for epistemic uncertainty and mean predictions
        openai_epistemic_uncertainty_over_time.append(openai_epistemic_uncertainty)
        meta_epistemic_uncertainty_over_time.append(meta_epistemic_uncertainty)
        openai_mean_predictions_over_time.append(openai_mean_predictions)
        meta_mean_predictions_over_time.append(meta_mean_predictions)

        # Store the results for each method
        accuracies_openai.append(acc_openai)
        accuracies_meta.append(acc_meta)
        accuracies_agg.append(acc_agg)
        accuracies_avg.append(acc_avg)

        f1_scores_openai.append(f1_openai)
        f1_scores_meta.append(f1_meta)
        f1_scores_agg.append(f1_agg)
        f1_scores_avg.append(f1_avg)

        log_likelihoods_openai.append(log_likelihood_openai)
        log_likelihoods_meta.append(log_likelihood_meta)
        log_likelihoods_agg.append(log_likelihood_agg)
        log_likelihoods_avg.append(log_likelihood_avg)

        ## Uncertainty analysis: 
        # Find indices where one model is correct and the other is incorrect
        correct_openai_incorrect_meta, correct_meta_incorrect_openai = identify_discrepancies(openai_mean_predictions, meta_mean_predictions, openai_ground_truths)

        # Analyse how often the LOWER uncertainty prediction is selected and correct
        selected_openai_confidence = compare_confidence(correct_openai_incorrect_meta, openai_mean_predictions, openai_mean_predictions, openai_epistemic_uncertainty, meta_epistemic_uncertainty, openai_ground_truths, eval_by='uncertainty')
        selected_meta_confidence = compare_confidence(correct_meta_incorrect_openai, meta_mean_predictions, meta_mean_predictions, openai_epistemic_uncertainty, meta_epistemic_uncertainty, openai_ground_truths, eval_by='uncertainty')
        selected_confidence_openai_over_time.append(selected_openai_confidence)
        selected_confidence_meta_over_time.append(selected_meta_confidence)

        # How often the OTHER model's prediction would have been better
        improve_with_openai, improve_with_meta = analyze_improvement(correct_openai_incorrect_meta, correct_meta_incorrect_openai, P_combined, openai_ground_truths)
        improved_selection_openai_over_time.append(improve_with_openai)
        improved_selection_meta_over_time.append(improve_with_meta)

        # Store ground truths
        gcs_over_time.append(openai_ground_truths)


    # Plot performance curves for each metric
    months = np.arange(0, len(prediction_files_openai))

    # Plot population
    plot_distribution_over_time(gcs_over_time, months, 
                                "Population Growth over Time", "Month", ylabel='Population', type='hist')

    # Plot uncertainty distributions over time
    plot_distribution_over_time(meta_epistemic_uncertainty_over_time, months, 
                                f"{args.models[1]} Model Epistemic Uncertainty Development", "Epistemic Uncertainty")
    # convert to confidence
    confidence_nn_over_time = [np.where(x > 0.5, x, 1 - x) for x in meta_mean_predictions_over_time]
    plot_distribution_over_time(confidence_nn_over_time, months, 
                                f"{args.models[1]} Model Confidence Development", "Confidence")
    
    plot_distribution_over_time(openai_epistemic_uncertainty_over_time, months, 
                                f"{args.models[0]} Model Uncertainty Development", "Epistemic Uncertainty")
    confidence_llm_over_time = [np.where(x > 0.5, x, 1 - x) for x in openai_mean_predictions_over_time]
    plot_distribution_over_time(confidence_llm_over_time, months, 
                                "LLM Confidence Development", "Confidence")
    
    # Plot uncertainty analysis over time 
    plot_uncertainties_vs_month(months, selected_confidence_openai_over_time, selected_confidence_meta_over_time, 
                                "Correct Lower Uncertainty Selections",
                                "correct_selections")
    plot_uncertainties_vs_month(months, improved_selection_openai_over_time, improved_selection_meta_over_time, 
                                "Potential Aggregation Improvements",
                                "aggregation_improvements")
    

    plot_performance_vs_month(months, accuracies_openai, openai_epistemic_uncertainty_over_time, accuracies_meta, meta_epistemic_uncertainty_over_time, accuracies_agg, accuracies_avg, "Accuracy")
    plot_performance_vs_month(months, f1_scores_openai, openai_epistemic_uncertainty_over_time, f1_scores_meta, meta_epistemic_uncertainty_over_time, f1_scores_agg, f1_scores_avg, "F1 Score")
    plot_performance_vs_month(months, log_likelihoods_openai, openai_epistemic_uncertainty_over_time, log_likelihoods_meta, meta_epistemic_uncertainty_over_time, log_likelihoods_agg, log_likelihoods_avg, "Log Likelihood")