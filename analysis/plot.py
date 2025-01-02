# imports 
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# from models.LLM_simulator import logistic_growth --> fix import

def logistic_growth(t, initial_mothers, L, k, t0):
    """Adjusted logistic growth model that starts with the initial number of mothers."""
    # Logistic growth model adds to the initial number of mothers
    return initial_mothers + (L - initial_mothers) / (1 + np.exp(-k * (t - t0)))


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


def plot_performance_vs_month_new(months, *metrics, metric_agg, metric_avg, metric_name, model_labels=None):
    """
    Plot performance metric vs. month for multiple models, including aggregated and averaged metrics,
    using a gradient of blues for each model.

    Args:
    - months: List of months.
    - *metrics: Metric values for each model (passed as separate arrays).
    - metric_agg: Metric values for aggregated predictions.
    - metric_avg: Metric values for direct averaging predictions.
    - metric_name: Name of the metric (e.g., 'Accuracy', 'F1 Score', 'Log Likelihood').
    - model_labels: List of labels for each model (default is 'Model 1', 'Model 2', ...).
    """
    plt.figure(figsize=(10, 6))

    # If no custom labels provided, create default labels
    if model_labels is None:
        model_labels = [f"Model {i+1}" for i in range(len(metrics))]

    # Set up a colormap for distinguishable shades of blue (skip the lightest shades)
    cmap = cm.get_cmap('Blues')
    colors = [cmap(0.5 + 0.3 * i / (len(metrics) - 1)) for i in range(len(metrics))]

    # Plot each model's metric with a different shade of blue
    for metric, label, color in zip(metrics, model_labels, colors):
        plt.plot(months, metric, label=label, marker='o', linestyle='--', color=color, alpha=0.7)

    # Plot Direct Averaging performance in orange
    if any(metric_avg):
        plt.plot(months, metric_avg, label='Posterior (Avg)', marker='o', color='orchid')

    # Plot Aggregated performance in green
    if any(metric_agg):
        plt.plot(months, metric_agg, label='Uncertainty-weighted Posterior', marker='o', linewidth=2.5, color='orange')

    plt.xlabel("Week Since Program Start")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} vs. Week")
    plt.legend()
    plt.grid(True)
    
    # Save the figure
    plt.savefig(f"{metric_name.replace(' ', '_').lower()}_vs_week.png")
    plt.show()


def plot_distribution_over_time(data, months, title, xlabel, ylabel='Density', type='kde'):
    """
    Plot the distribution epistemic uncertainty / avg probability over time.
    
    Args:
    - data: A list of arrays, where each array contains the values for a particular time step.
    - months: List of time steps.
    - title: Title of the plot.
    - xlabel: Label for x-axis.
    """

    if type == 'kde':
        plt.figure(figsize=(10, 6))
        colours = sns.color_palette("Oranges_r", len(months))

        for i, month_data in enumerate(data):
            sns.kdeplot(month_data, label=f'Month {months[i]}', fill=True, color=colours[i], clip=(0, 1))
                    #, cumulative=True)

        plt.legend(loc='best', title="Months")
    
    elif type == 'hist':
        months = np.arange(1, len(months) + 1)

        total_counts = [len(month) for month in data]
        engaged = [sum(month) for month in data]
        not_engaged = [len(month)-sum(month) for month in data]
        proportion_engaged = [e / t for e, t in zip(engaged, total_counts)]
        proportion_not_engaged = [ne / t for ne, t in zip(not_engaged, total_counts)]

        bar_width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))

        # ax.bar(months - bar_width/2, not_engaged, bar_width, label='not engaged')
        # ax.bar(months + bar_width/2, engaged, bar_width, label='engaged')
        ax.bar(months, total_counts, label='Total counts')
        x = np.arange(1, 22)
        xs = np.arange(0, 22)
        ax.plot(xs, [15] + [logistic_growth(t, 15, 3000, 0.4, 10) for t in x], label='Logistic Function', color='orange')

        for i in range(len(months)):
            # ax.text(months[i] - bar_width/2, not_engaged[i] + 0.05, 
            #         f'{proportion_not_engaged[i]:.2f}', ha='center', va='bottom')
            # ax.text(months[i] + bar_width/2, engaged[i] + 0.05, 
            #         f'{proportion_engaged[i]:.2f}', ha='center', va='bottom')
            ax.text(months[i], total_counts[i] + 0.05, 
                    f'{(total_counts[i])}', ha='center', va='bottom')

        ax.set_xticks(xs)
        ax.legend(loc='best')
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()


def plot_uncertainties_vs_month(months, llm_values, mc_values, plot_title, plot_name):
    """
    Plot results over time for a given question.
    
    Args:
    - months: List of month indices.
    - llm_values: List of LLM results over time.
    - mc_values: List of MC results over time.
    - question_name: Name of the question being analyzed (for plot title).
    - plot_name: Name of the plot file to save (str)
    """

    plt.figure(figsize=(10, 6))
    plt.plot(months, llm_values, label='OpenAI correct', marker='o')
    plt.plot(months, mc_values, label='Anthropic correct', marker='o')
    
    plt.xlabel('Months')
    plt.ylabel('Percentage')
    plt.title(f'{plot_title} Over Time')
    plt.legend()
    plt.grid(True)

    plt.savefig(plot_name + ".png")

    plt.show()


def plot_k_corresponding(models, ground_truth, k, metric_name="Correctly Predicted Lowest-k Engagement"):
    """
    Generate a bar chart showing the percentage of correctly predicted lowest-k engagement beneficiaries 
    for each model overall.

    Args:
    - models: Dictionary with model names as keys and their binary engagement data as values.
    - ground_truth: List of binary ground truth arrays for engagement (0s and 1s).
    - k: Number of lowest engagement beneficiaries to consider.
    - metric_name: Name of the metric to display in the chart title.
    """
    # Combine all ground truth arrays into a single array
    combined_ground_truth = np.concatenate(ground_truth)

    # Identify indices of the ground truth's lowest-k engagement beneficiaries
    true_lowest_k_indices = set(np.argsort(combined_ground_truth)[:k])

    # Calculate accuracy for each model
    model_accuracies = {}
    for model_name, predictions_list in models.items():
        # Combine all prediction arrays for the current model
        combined_predictions = np.concatenate(predictions_list)

        # Identify the predicted lowest-k engagement beneficiaries
        predicted_lowest_k_indices = set(np.argsort(combined_predictions)[:k])
        
        # Compute accuracy
        correct_count = len(predicted_lowest_k_indices & true_lowest_k_indices)
        accuracy = (correct_count / k) * 100  # Percentage of correct predictions
        model_accuracies[model_name] = accuracy

    # Plot the bar chart
    plt.figure(figsize=(10, 6))
    model_names = list(model_accuracies.keys())
    accuracies = list(model_accuracies.values())
    bar_positions = np.arange(len(model_names))

    plt.bar(bar_positions, accuracies, color='skyblue', edgecolor='black')
    plt.xticks(bar_positions, model_names, rotation=45, ha='right')
    plt.xlabel("Models")
    plt.ylabel("Accuracy (%)")
    plt.title(f"{metric_name} (k={k})")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Annotate bar values
    for i, accuracy in enumerate(accuracies):
        plt.text(i, accuracy + 1, f"{accuracy:.1f}%", ha='center', va='bottom', fontsize=10)

    # Save and show plot
    plt.tight_layout()
    plt.savefig(f"overall_lowest_k_engagement_accuracy_k_{k}.png")
    plt.show()