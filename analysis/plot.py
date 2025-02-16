# imports 
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


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
        cmap = matplotlib.colormaps['Paired']  
        colors = [cmap(i) for i in range(len(metrics))]

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
                        marker='o', markersize=3, color='grey') 

            # Plot aggregated performance
            if any(metric_agg):
                ax.plot(months, metric_agg, label='Uncertainty-weighted Aggregation', 
                        marker='o', markersize=3, color='#DC143C') 

            # Subplot config
            ax.set_ylabel(metric_name, fontsize=12)
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


def plot_accuracy_by_feature(df, features=None, metric='accuracy', 
                             models=None, model_labels=None, figsize=(10, 3)):
    
    if features is None: # default to all features in df
        features = df['feature'].unique()  

    if models is None: # default to all models in df
        models = df['model'].unique() 

    if model_labels is None: # default to model names
        model_labels = models

    # set up axes
    num_features = len(features)
    num_rows = math.ceil(num_features/2) 
    fig, axes = plt.subplots(num_rows, 2, figsize=(figsize[0], figsize[1] * num_rows), sharey=True)
    axes = axes.flatten() 

    bar_width = 0.118

    for ax, feature in zip(axes, features):

        feature_df = df[df['feature'] == feature] 
        feature_categories = feature_df['category'].unique()
        if feature == 'income':
            feature_categories = [c.replace('income_', '') for c in feature_categories]
        elif feature == 'age':
            feature_categories = [c.replace('age_', '') for c in feature_categories]
        elif feature == 'education':
            feature_categories = [c.replace('education_', '') for c in feature_categories]
        elif feature == 'language':
            feature_categories = [c.replace('language_', '') for c in feature_categories]

        x = np.arange(len(feature_categories))  # mid x pos for categories

        cmap = matplotlib.colormaps['Paired']  
        colors = [cmap(i) for i in range(12)] 
        model_colors = [colors[i] for i in [0,1,2,3,7]] + ['#DC143C', 'k', 'gray']
        
        for i, model in enumerate(models):
            model_df = feature_df[feature_df['model'] == model]
            if metric == 'log_likelihood':
                to_plot = model_df.groupby('category')[metric].mean()
            else: 
                to_plot = model_df.groupby('category')[metric].mean()
            std_dev = model_df.groupby('category')[metric + '_std'].mean()
            
            # diff x positions for each model
            x_positions = x + i * bar_width

            # plot bars w error bars
            bars = ax.bar(x_positions, to_plot.values, 
                          yerr=std_dev.values, capsize=bar_width*20, error_kw={'elinewidth': 0.1, 'alpha': 0.5}, #, 'color': '#3b3b3b'},
                          width=bar_width, label=model_labels[i], alpha=0.5, 
                          color=model_colors[i])#, edgecolor='black')

            # # Print mean accuracy 
            # for bar in bars:
            #     height = bar.get_height()
            #     ax.text(bar.get_x() + bar.get_width() / 2, height+0.09, f'{height:.1f}', 
            #             ha='center', va='bottom', fontsize=9, color='black', rotation=90)

        ax.set_title(f'{metric.capitalize()} by {feature.capitalize()}', fontsize=14)
        ax.set_xlabel(feature.capitalize(), fontsize=12)
        ax.set_xticks(x + bar_width * (len(models) - 1) / 2) 
        ax.set_xticklabels([f.capitalize() for f in feature_categories], rotation=10, fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Remove extra empty axes
    for ax in axes[num_features:]:
        ax.set_visible(False)

    axes[0].set_ylabel(metric.capitalize(), fontsize=12)
    axes[2].set_ylabel(metric.capitalize(), fontsize=12) 
    fig.suptitle(f'Model {metric.capitalize()} by Feature', fontsize=16, y=0.99)
    fig.legend(model_labels,
            loc='upper center',
            bbox_to_anchor=(0.5, 0.97), 
            ncol=len(models)//2, 
            fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95]) 
    plt.savefig(f'plots/bias_by_feature_{metric}.png')
    plt.show()


def plot_engagement_over_time(models, model_labels, model_results, ground_truths, labels=None):

    # mean engagement per week for ground truths --> transpose if shape is (num_weeks, num_mothers)
    if ground_truths.shape[0] < ground_truths.shape[1]: 
        ground_truths = ground_truths.T
    gcs = np.array(ground_truths)  # (num_mothers, num_weeks)
    engagement_over_time = np.mean(gcs, axis=0) 

    # Plot engagement over time
    plt.figure(figsize=(15, 6))

    cmap = matplotlib.colormaps['Paired']
    colors = [cmap(i) for i in range(12)]
    colors = [colors[i] for i in [0,1,2,3,6,7,8,9,4,5,10,11]]

    # Plot Ground Truth Engagement
    plt.plot(range(1, len(engagement_over_time) + 1), engagement_over_time, 
            marker="o", markersize=5, linestyle="-", color="black", label="Ground Truth Engagement")

    # Plot predicted engagement for each model
    for i, model in enumerate(models):
        # mean engagement per week
        model_predictions = np.array(model_results[model]["mean_predictions"])  # (num_time_steps, num_mothers)
        mean_engagement_predictions = np.mean(model_predictions, axis=1)  # avg  over all mothers

        plt.plot(range(1, len(mean_engagement_predictions) + 1), mean_engagement_predictions, 
                    marker="o", linestyle="--", markersize=4, color=colors[i],
                    label=f"{model_labels[i]} Predictions")


    plt.title("Mean enagement over time")
    plt.xlabel("Weeks")
    plt.ylabel("Proportion Engaged")
    plt.legend()
    plt.grid(True)
    plt.savefig("engagement_over_time.png")
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


# def plot_performance_vs_month(months, metric_llm, uncertainty_llm, metric_nn, uncertainty_nn, metric_agg, metric_avg, metric_name):
#     """
#     Plot performance metric vs. month and save the figure.

#     Args:
#     - months: List of months.
#     - metric_llm: Metric values for LLM predictions.
#     - metric_nn: Metric values for NN predictions.
#     - metric_agg: Metric values for aggregated predictions.
#     - metric_avg: Metric values for direct averaging predictions.
#     - metric_name: Name of the metric (e.g., 'Accuracy', 'F1 Score', 'Log Likelihood').
#     """
#     plt.figure(figsize=(10, 6))
    
#     # Plot LLM performance
#     plt.plot(months, metric_llm, label='OpenAI Model', marker='o', linestyle='--')
    
#     # Plot NN performance
#     if any(metric_nn):
#         plt.plot(months, metric_nn, label='Anthropic Model', marker='o', linestyle='--')
    
#     # Plot Direct Averaging performance
#     if any(metric_avg):
#         plt.plot(months, metric_avg, label='Posterior', marker='o')

#     # Plot Aggregated performance
#     if any(metric_agg):
#         plt.plot(months, metric_agg, label='Uncertainty-weighted Posterior', marker='o', linewidth=2.5)

#     plt.xlabel("Month")
#     plt.ylabel(metric_name)
#     plt.title(f"{metric_name} vs. Month")
#     plt.legend()
#     plt.grid(True)
    
#     # Save the figure
#     plt.savefig(f"{metric_name.replace(' ', '_').lower()}_vs_month.png")
    
#     plt.show()