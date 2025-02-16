# imports 
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_performance_vs_month_new(months, *metrics, 
                                metric_agg, metric_avg, metric_low,
                                metric_name, model_labels=None):
    if model_labels is None:
        model_labels = [f"Model {i+1}" for i in range(len(metrics))]

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
    colors = [colors[i] for i in [0,1,2,3,7,6,8,9,4,5,10,11]]
    labels = ['Google', 'OpenAI', 'Anthropic']

    # Variables to store line objects for the common legend
    line_avg, line_low, line_agg = None, None, None

    for idx in range(num_pairs):
        ax = axes[idx]

        model1_idx = idx * 2
        metric1 = metrics[model1_idx]
        label1 = model_labels[model1_idx]
        ax.plot(months, metric1, label=label1, marker='o', markersize=3, 
                linestyle='-.', color=colors[model1_idx], alpha=1)

        if model1_idx < len(metrics)-1:
            model2_idx = min(model1_idx + 1, len(metrics) - 1)  
            metric2 = metrics[model2_idx]
            label2 = model_labels[model2_idx]
            ax.plot(months, metric2, label=label2, marker='o', markersize=3, 
                    linestyle='-.', color=colors[model2_idx], alpha=1)

        # Plot common lines without labels (except first plot to get line objects)
        if idx == 0:
            if any(metric_low):
                line_low, = ax.plot(months, metric_low, marker='o', markersize=3, 
                                  color='grey', label='_nolegend_')
            if any(metric_avg):
                line_avg, = ax.plot(months, metric_avg, marker='o', markersize=3, 
                                  color='k', label='_nolegend_')
            if any(metric_agg):
                line_agg, = ax.plot(months, metric_agg, marker='o', markersize=3, 
                                  color='#DC143C', label='_nolegend_')
        else:
            if any(metric_low):
                ax.plot(months, metric_low, marker='o', markersize=3, 
                       color='grey', label='_nolegend_')
            if any(metric_avg):
                ax.plot(months, metric_avg, marker='o', markersize=3, 
                       color='k', label='_nolegend_')
            if any(metric_agg):
                ax.plot(months, metric_agg, marker='o', markersize=3, 
                       color='#DC143C', label='_nolegend_')

        # Subplot config
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(f"{labels[idx]} Models", fontsize=14)
        ax.legend(fontsize=12, loc='upper right')  
        ax.grid(True, alpha=0.4)
    axes[-1].set_xlabel("Weeks Since Program Start", fontsize=12)

    # Hide unused subplots
    for idx in range(num_pairs, len(axes)):
        axes[idx].set_visible(False)

    # Add the common lines legend at the top
    legend_elements = []
    legend_labels = []
    if line_low is not None:
        legend_elements.append(line_low)
        legend_labels.append("Lowest Uncertainty")
    if line_avg is not None:
        legend_elements.append(line_avg)
        legend_labels.append("Direct Averaging")
    if line_agg is not None:
        legend_elements.append(line_agg)
        legend_labels.append("Uncertainty-weighted Aggregation")

    # Add top legend
    if legend_elements:
        fig.legend(legend_elements, legend_labels,
                  loc='upper center', bbox_to_anchor=(0.5, 0.99),
                  ncol=len(legend_elements), fontsize=12)
        fig.subplots_adjust(top=0.85)

    # fig.suptitle(f'', fontsize=16, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95]) 
    plt.savefig(f"plots/{metric_name.replace(' ', '_').lower()}_vs_week_subplots.png")
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


def plot_engagement_over_time(models, model_labels, model_results, 
                              ground_truths, agg, avg, lowest_unc):
    """
    Plot mean engagement vs time in subplots (pairs of models per subplot),
    alongside ground truth, direct averaging, lowest uncertainty, and 
    uncertainty-weighted aggregation.
    """

    # Compute mean engagement for ground truth
    if ground_truths.shape[0] < ground_truths.shape[1]:
        ground_truths = ground_truths.T
    gcs = np.array(ground_truths)  
    engagement_over_time = np.mean(gcs, axis=0)  

    # Compute mean engagement (and std) for each ensemble approach
    agg_mean = np.mean(np.array(agg), axis=1)
    avg_mean = np.mean(np.array(avg), axis=1)
    low_mean = np.mean(np.array(lowest_unc), axis=1)

    # Prepare subplots for adjacent pairs of models
    num_pairs = (len(models) + 1) // 2
    fig, axes = plt.subplots(num_pairs, 1, figsize=(10, 3 * num_pairs), 
                             sharex=True, sharey=False)
    if num_pairs == 1:
        axes = [axes]  # Ensure axes is iterable if there's only one pair

    cmap = matplotlib.colormaps['Paired']
    colors = [cmap(i) for i in range(12)]
    colors = [colors[i] for i in [0,1,2,3,7,6,8,9,4,5,10,11]]
    labels = ['Google', 'OpenAI', 'Anthropic']

    line_low, line_avg, line_agg, line_gc = None, None, None, None

    for idx in range(num_pairs):
        ax = axes[idx]

        # Plot ensemble approaches with error bands
        if idx == 0:
            x = range(1, len(low_mean)+1)
            
            # Lowest Uncertainty
            line_low, = ax.plot(x, low_mean, marker='o', markersize=3, 
                                color='gray', label="_nolegend_")
            
            # Direct Averaging
            line_avg, = ax.plot(x, avg_mean, marker='o', markersize=3, 
                            color='k', label="_nolegend_")
            
            # Uncertainty-weighted Aggregation
            line_agg, = ax.plot(x, agg_mean, marker='o', markersize=3, 
                            color='#DC143C', label="_nolegend_")
            
            # Ground Truth
            line_gc, = ax.plot(x, engagement_over_time, marker="o", markersize=3, 
                            color="purple", label="_nolegend_")
        else:
            ax.plot(range(1, len(low_mean)+1), low_mean, 
                    marker='o', markersize=3, color='gray', label="_nolegend_")
            ax.plot(range(1, len(avg_mean)+1), avg_mean, 
                    marker='o', markersize=3, color='k', label="_nolegend_")
            ax.plot(range(1, len(agg_mean)+1), agg_mean, 
                    marker='o', markersize=3, color='#DC143C', label="_nolegend_")
            ax.plot(range(1, len(engagement_over_time) + 1),
                    engagement_over_time, marker="o", markersize=3, 
                    linestyle="-", color="purple", label="_nolegend_")

        # Plot up to 2 models in each subplot
        model1_idx = idx * 2
        model2_idx = model1_idx + 1

        # First model in the pair
        if model1_idx < len(models):
            model_predictions = np.array(model_results[models[model1_idx]]["mean_predictions"])
            mean_pred = np.mean(model_predictions, axis=1)            
            x = range(1, len(mean_pred) + 1)
            ax.plot(x, mean_pred, 
                    marker="o", linestyle="-.", markersize=3, 
                    color=colors[model1_idx], label=f"{model_labels[model1_idx]}")

        # Second model in the pair
        if model2_idx < len(models):
            model_predictions = np.array(model_results[models[model2_idx]]["mean_predictions"])
            mean_pred = np.mean(model_predictions, axis=1)
            ax.plot(x, mean_pred,
                    marker="o", linestyle="-.", markersize=3, 
                    color=colors[model2_idx], label=f"{model_labels[model2_idx]}")

        ax.set_ylabel("Proportion Engaged")
        ax.set_title(labels[idx] + " Models")
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True)

    axes[-1].set_xlabel("Weeks")
    # fig.suptitle(f'', fontsize=16, y=0.99)
    fig.subplots_adjust(top=0.85)
    fig.legend([line_low, line_avg, line_agg, line_gc],
               ["Lowest Uncertainty", "Direct Averaging", "Uncertainty-weighted Aggregation", "Ground Truth"],
               loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=4, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95]) 
    plt.savefig("plots/engagement_over_time_subplots.png")
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