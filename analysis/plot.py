# imports 
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_performance_vs_month(months, metric_llm, uncertainty_llm, metric_nn, uncertainty_nn, metric_agg, metric_avg, metric_name):
    """
    Plot performance metric vs. month and save the figure.

    Args:
    - months: List of months.
    - metric_llm: Metric values for LLM predictions.
    - metric_nn: Metric values for NN predictions.
    - metric_agg: Metric values for aggregated predictions.
    - metric_avg: Metric values for direct averaging predictions.
    - metric_name: Name of the metric (e.g., 'Accuracy', 'F1 Score', 'Log Likelihood').
    """
    plt.figure(figsize=(10, 6))
    
    # Plot LLM performance
    plt.plot(months, metric_llm, label='OpenAI Model', marker='o', linestyle='--')
    
    # Plot NN performance
    if any(metric_nn):
        plt.plot(months, metric_nn, label='Anthropic Model', marker='o', linestyle='--')
    
    # Plot Direct Averaging performance
    if any(metric_avg):
        plt.plot(months, metric_avg, label='Posterior', marker='o')

    # Plot Aggregated performance
    if any(metric_agg):
        plt.plot(months, metric_agg, label='Uncertainty-weighted Posterior', marker='o', linewidth=2.5)

    plt.xlabel("Month")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} vs. Month")
    plt.legend()
    plt.grid(True)
    
    # Save the figure
    plt.savefig(f"{metric_name.replace(' ', '_').lower()}_vs_month.png")
    
    plt.show()


def plot_performance_vs_month_new(months, *metrics, 
                                  metric_agg, metric_avg, metric_low,
                                  metric_name, model_labels=None, separate_axes=False):
    """
    Plot performance metric vs. month for multiple models and baselines, 
    with an option to use subplots for adjacent pairs of models.

    Args:
    - months: List of months.
    - *metrics: Metric values for each model (passed as separate arrays).
    - metric_agg: Metric values for aggregated predictions.
    - metric_avg: Metric values for direct averaging predictions.
    - metric_low: Metric values for lowest uncertainty model.
    - metric_name: Name of the metric (e.g., 'Accuracy', 'F1 Score', 'Log Likelihood').
    - model_labels: List of labels for each model (default is 'Model 1', 'Model 2', ...).
    - separate_axes: If True, create subplots for adjacent pairs of models.
    """

    if model_labels is None:
        model_labels = [f"Model {i+1}" for i in range(len(metrics))]

    if not separate_axes:
        # Plot all models on the same axes

        plt.figure(figsize=(10, 6))
        cmap = matplotlib.colormaps['Paired']  # Use a categorical colormap
        colors = [cmap(i / (len(metrics))) for i in range(len(metrics))]

        # Plot metrics for each model
        for metric, label, color in zip(metrics, model_labels, colors):
            plt.plot(months, metric, label=label, marker='o', linestyle='..', color=color, alpha=0.7)

        # Plot direct average performance
        if any(metric_avg):
            plt.plot(months, metric_avg, label='Direct Averaging', marker='o', color='purple')

        # Plot lowest uncertainty model performance
        if any(metric_low):
            plt.plot(months, metric_low, label='Lowest Uncertainty Model', marker='o', color='#DC143C')

        # Plot aggregated performance
        if any(metric_agg):
            plt.plot(months, metric_agg, label='Uncertainty-weighted Aggregation', marker='o', linewidth=2.5, color='g')

        plt.xlabel("Week Since Program Start")
        plt.ylabel(metric_name)
        plt.title(f"{metric_name} vs. Week")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{metric_name.replace(' ', '_').lower()}_vs_week.png")
        plt.show()

    else:
        # Create subplots for adjacent pairs of models
        num_pairs = (len(metrics) + 1) // 2 
        num_cols = 1
        num_rows = math.ceil(num_pairs / num_cols)

        fig, axes = plt.subplots(num_rows, num_cols, 
                                 figsize=(10, 3 * num_rows), 
                                 sharey=False, sharex=True)
        axes = axes.flatten()
        cmap = matplotlib.colormaps['Paired']
        colors = [cmap(i) for i in range(12)]
        colors = [colors[i] for i in [0,1,2,3,6,7,8,9,4,5,10,11]]
        labels = ['Google', 'OpenAI', 'Anthropic']

        for idx in range(num_pairs):
            ax = axes[idx]

            model1_idx = idx * 2
            metric1 = metrics[model1_idx]
            label1 = model_labels[model1_idx]
            ax.plot(months, metric1, label=label1, marker='o', markersize=3, linestyle='-.', color=colors[model1_idx], alpha=1)

            if model1_idx <= len(metrics)-1: # else odd number of models
                model2_idx = min(model1_idx + 1, len(metrics) - 1)  
                metric2 = metrics[model2_idx]
                label2 = model_labels[model2_idx]
                ax.plot(months, metric2, label=label2, marker='o', markersize=3, linestyle='-.', color=colors[model2_idx], alpha=1)

            # Plot direct average performance
            if any(metric_avg):
                ax.plot(months, metric_avg, label='Direct Averaging', 
                        marker='o', markersize=3, color='k')

            # Plot lowest uncertainty model performance
            if any(metric_low):
                ax.plot(months, metric_low, label='Lowest Uncertainty', 
                        marker='o', markersize=3, color='grey') #'#DC143C')

            # Plot aggregated performance
            if any(metric_agg):
                ax.plot(months, metric_agg, label='Uncertainty-weighted Aggregation', 
                        marker='o', markersize=3, color='#DC143C') #'orange')

            # Subplot config
            ax.set_ylabel(metric_name, fontsize=12)
            # ax.set_title(f"{metric_name} of {label1} vs {label2} - ")
            ax.set_title(f"{labels[idx]}", fontsize=14)
            ax.legend(fontsize=12, loc='lower left')
            ax.grid(True, alpha=0.4)
        axes[-1].set_xlabel("Weeks Since Program Start", fontsize=12)
        # axes[-1].bbox_to_anchor = (1, 1)

        # Hide unused subplots
        for idx in range(num_pairs, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        plt.savefig(f"{metric_name.replace(' ', '_').lower()}_vs_week_subplots.png")
        plt.show()


def plot_uncertainty_over_time(timesteps, model_results, 
                               combined_uncertainty, 
                               direct_avg_uncertainty, 
                               lowest_uncertainty,
                               models):
    """
    Plot epistemic uncertainty over time for each model, aggregated model, and baselines.

    Args:
    - timesteps: List of timesteps (e.g., weeks or months).
    - model_results: Dictionary containing epistemic uncertainty for each model.
    - combined_uncertainty: Epistemic uncertainty for the aggregated model.
    - direct_avg_uncertainty: Epistemic uncertainty for the direct average model.
    - lowest_uncertainty: Epistemic uncertainty for the lowest uncertainty model.
    - models: List of model names to include in the plot.
    """
    plt.figure(figsize=(12, 8))
    cmap = matplotlib.colormaps['Paired']
    colors = [cmap(i / (len(models))) for i in range(len(models))]

    # model uncertainties
    for model, color in zip(models, colors):
        model_uncertainty = [np.mean(unc) for unc in model_results[model]["epistemic_uncertainty"]]
        plt.plot(timesteps, model_uncertainty, label=f"{model} Uncertainty", marker='o', linestyle='--', color=color)

    # direct average model uncertainty 
    direct_avg_uncertainty = [np.mean(unc) for unc in direct_avg_uncertainty]
    plt.plot(timesteps, direct_avg_uncertainty, label="Direct Average Model Uncertainty", marker='o', linestyle='-', color='purple')

    # lowest-uncertainty
    lowest_uncertainty = [np.mean(unc) for unc in lowest_uncertainty]
    plt.plot(timesteps, lowest_uncertainty, label="Lowest Uncertainty Selection", marker='o', linestyle='-', color='#DC143C')

    # aggregated model uncertainty
    combined_uncertainty = [np.mean(unc) for unc in combined_uncertainty]
    plt.plot(timesteps, combined_uncertainty, label="Aggregated Model Uncertainty", marker='o', linestyle='-', linewidth=2.5, color='g')

    plt.xlabel("Weeks Since Program Start", fontsize=12)
    plt.ylabel("Epistemic Uncertainty", fontsize=12)
    plt.title("Epistemic Uncertainty Over Time", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('epistemic_uncertainty_over_time.png')
    plt.show()


# def logistic_growth(t, a, b, k, m):
#     return a + (b - a) / (1 + k * np.exp(-m * t))


# def plot_distribution_over_time(data, months, title, xlabel, ylabel='Density', type='kde'):
#     """
#     Plot the distribution epistemic uncertainty / avg probability over time.
    
#     Args:
#     - data: A list of arrays, where each array contains the values for a particular time step.
#     - months: List of time steps.
#     - title: Title of the plot.
#     - xlabel: Label for x-axis.
#     """

#     if type == 'kde':
#         plt.figure(figsize=(10, 6))
#         colours = sns.color_palette("Oranges_r", len(months))

#         for i, month_data in enumerate(data):
#             sns.kdeplot(month_data, label=f'Month {months[i]}', fill=True, color=colours[i], clip=(0, 1))
#                     #, cumulative=True)

#         plt.legend(loc='best', title="Months")
    
#     elif type == 'hist':
#         months = np.arange(1, len(months) + 1)

#         total_counts = [len(month) for month in data]
#         engaged = [sum(month) for month in data]
#         not_engaged = [len(month)-sum(month) for month in data]
#         proportion_engaged = [e / t for e, t in zip(engaged, total_counts)]
#         proportion_not_engaged = [ne / t for ne, t in zip(not_engaged, total_counts)]

#         bar_width = 0.35

#         fig, ax = plt.subplots(figsize=(10, 6))

#         # ax.bar(months - bar_width/2, not_engaged, bar_width, label='not engaged')
#         # ax.bar(months + bar_width/2, engaged, bar_width, label='engaged')
#         ax.bar(months, total_counts, label='Total counts')
#         x = np.arange(1, 22)
#         xs = np.arange(0, 22)
#         ax.plot(xs, [15] + [logistic_growth(t, 15, 3000, 0.4, 10) for t in x], label='Logistic Function', color='orange')

#         for i in range(len(months)):
#             # ax.text(months[i] - bar_width/2, not_engaged[i] + 0.05, 
#             #         f'{proportion_not_engaged[i]:.2f}', ha='center', va='bottom')
#             # ax.text(months[i] + bar_width/2, engaged[i] + 0.05, 
#             #         f'{proportion_engaged[i]:.2f}', ha='center', va='bottom')
#             ax.text(months[i], total_counts[i] + 0.05, 
#                     f'{(total_counts[i])}', ha='center', va='bottom')

#         ax.set_xticks(xs)
#         ax.legend(loc='best')
    
#     plt.title(title)
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.grid(True)
#     plt.show()


# def plot_uncertainties_vs_month(months, llm_values, mc_values, plot_title, plot_name):
#     """
#     Plot results over time for a given question.
    
#     Args:
#     - months: List of month indices.
#     - llm_values: List of LLM results over time.
#     - mc_values: List of MC results over time.
#     - question_name: Name of the question being analyzed (for plot title).
#     - plot_name: Name of the plot file to save (str)
#     """

#     plt.figure(figsize=(10, 6))
#     plt.plot(months, llm_values, label='OpenAI correct', marker='o')
#     plt.plot(months, mc_values, label='Anthropic correct', marker='o')
    
#     plt.xlabel('Months')
#     plt.ylabel('Percentage')
#     plt.title(f'{plot_title} Over Time')
#     plt.legend()
#     plt.grid(True)

#     plt.savefig(plot_name + ".png")

#     plt.show()


# def plot_k_corresponding(models, ground_truth, k, metric_name="Correctly Predicted Lowest-k Engagement"):
#     """
#     Generate a bar chart showing the percentage of correctly predicted lowest-k engagement beneficiaries 
#     for each model overall.

#     Args:
#     - models: Dictionary with model names as keys and their binary engagement data as values.
#     - ground_truth: List of binary ground truth arrays for engagement (0s and 1s).
#     - k: Number of lowest engagement beneficiaries to consider.
#     - metric_name: Name of the metric to display in the chart title.
#     """
#     # Combine all ground truth arrays into a single array
#     combined_ground_truth = np.concatenate(ground_truth)

#     # Identify indices of the ground truth's lowest-k engagement beneficiaries
#     true_lowest_k_indices = set(np.argsort(combined_ground_truth)[:k])

#     # Calculate accuracy for each model
#     model_accuracies = {}
#     for model_name, predictions_list in models.items():
#         # Combine all prediction arrays for the current model
#         combined_predictions = np.concatenate(predictions_list)

#         # Identify the predicted lowest-k engagement beneficiaries
#         predicted_lowest_k_indices = set(np.argsort(combined_predictions)[:k])
        
#         # Compute accuracy
#         correct_count = len(predicted_lowest_k_indices & true_lowest_k_indices)
#         accuracy = (correct_count / k) * 100  # Percentage of correct predictions
#         model_accuracies[model_name] = accuracy

#     # Plot the bar chart
#     plt.figure(figsize=(10, 6))
#     model_names = list(model_accuracies.keys())
#     accuracies = list(model_accuracies.values())
#     bar_positions = np.arange(len(model_names))

#     plt.bar(bar_positions, accuracies, color='skyblue', edgecolor='black')
#     plt.xticks(bar_positions, model_names, rotation=45, ha='right')
#     plt.xlabel("Models")
#     plt.ylabel("Accuracy (%)")
#     plt.title(f"{metric_name} (k={k})")
#     plt.grid(axis='y', linestyle='--', alpha=0.7)

#     # Annotate bar values
#     for i, accuracy in enumerate(accuracies):
#         plt.text(i, accuracy + 1, f"{accuracy:.1f}%", ha='center', va='bottom', fontsize=10)

#     # Save and show plot
#     plt.tight_layout()
#     plt.savefig(f"overall_lowest_k_engagement_accuracy_k_{k}.png")
#     plt.show()