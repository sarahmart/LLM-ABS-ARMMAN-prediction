# imports 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# from models.LLM_simulator import logistic_growth --> fix import

def logistic_growth(t, initial_mothers, L, k, t0):
    """Adjusted logistic growth model that starts with the initial number of mothers."""
    # Logistic growth model adds to the initial number of mothers
    return initial_mothers + (L - initial_mothers) / (1 + np.exp(-k * (t - t0)))


def plot_performance_vs_month(months, metric_llm, metric_nn, metric_agg, metric_avg, metric_name):
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
    plt.plot(months, metric_llm, label='LLM', marker='o')
    
    # Plot NN performance, but skip the first two months (since NN is None there)
    if any(metric_nn):
        plt.plot(months[2:], metric_nn[2:], label='Two-stage', marker='o')

    # Plot Direct Averaging performance, but skip the first two months (since Direct Averaging is None there)
    if any(metric_avg):
        plt.plot(months[2:], metric_avg[2:], label='Posterior', marker='o')

    # Plot Aggregated performance, but skip the first two months (since Aggregated is None there)
    if any(metric_agg):
        plt.plot(months[2:], metric_agg[2:], label='Posterior - weighted by UQ', marker='o')

    plt.xlabel("Month")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} vs. Month")
    plt.legend()
    plt.grid(True)
    
    # Save the figure
    plt.savefig(f"{metric_name.replace(' ', '_').lower()}_vs_month.png")
    
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
    plt.plot(months, llm_values, label='LLM correct', marker='o')
    plt.plot(months, mc_values, label='Two-stage correct', marker='o')
    
    plt.xlabel('Months')
    plt.ylabel('Percentage')
    plt.title(f'{plot_title} Over Time')
    plt.legend()
    plt.grid(True)

    plt.savefig(plot_name + ".png")

    plt.show()