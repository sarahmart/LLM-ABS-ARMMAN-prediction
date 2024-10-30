import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import numpy as np
import json
import matplotlib.pyplot as plt
import random
import os
from sklearn.metrics import f1_score  # Importing F1 score calculation


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class StatePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_prob=0.5):
        super(StatePredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        return x


def data_preprocessing(data_path):
    """Preprocess the data from the given JSON file."""
    with open(data_path, 'r') as f:
        data = json.load(f)

    groups = ['control']  # Specify the groups to process
    for group in groups:
        group_actions = np.array(data[group]['actions'])
        print(f'Group {group}: Avg pulls = {np.average(group_actions)}')

    num_arms = 3000  # Total number of arms
    num_steps = 39  # Number of time steps (weeks)
    print(f'Number of steps: {num_steps}')

    state_trajectories = [[] for _ in range(num_arms)]
    action_trajectories = [[] for _ in range(num_arms)]
    features = []
    i = 0
    for group in groups:
        group_actions = np.array(data[group]['actions'])
        group_states = np.array(data[group]['states'])
        group_features = np.array(data[group]['features']).astype(np.float32)  # Convert to float32
        features.extend(group_features)
        for arm in range(3000):
            for j in range(num_steps):
                state_trajectories[i].append(group_states[arm, j])
                action_trajectories[i].append(group_actions[arm, j])
            state_trajectories[i].append(group_states[arm, num_steps])
            i += 1

    return features, state_trajectories, action_trajectories


def train_model_with_early_stopping(model, criterion, optimizer, train_loader, val_loader, device, save_best, num_epochs=10, patience=5):
    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for features, state, action, next_state in train_loader:
            inputs = torch.cat((features.float(), state.unsqueeze(1).float(), action.unsqueeze(1).float()), dim=1).to(device)
            targets = next_state.unsqueeze(1).float().to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = outputs.round()
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

        val_loss = compute_validation_loss(model, criterion, val_loader, device)
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {running_loss / len(train_loader):.4f}, Validation Loss: {val_loss:.4f}")

        if save_best:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break
        else:
            best_model_state = model.state_dict().copy()

    model.load_state_dict(best_model_state)
    return model, best_val_loss


def compute_validation_loss(model, criterion, val_loader, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for features, state, action, next_state in val_loader:
            inputs = torch.cat((features.float(), state.unsqueeze(1).float(), action.unsqueeze(1).float()), dim=1).to(device)
            targets = next_state.unsqueeze(1).float().to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

    return val_loss / len(val_loader)


def apply_dropout(m):
    """Function to apply dropout during test time."""
    if type(m) == nn.Dropout:
        m.train()


def mc_dropout_predictions(model, dataloader, device, mc_iterations=10):
    """Perform MC-Dropout during test time to generate multiple predictions."""
    model.apply(apply_dropout)  # Enable dropout layers at test time
    all_mc_predictions = []  # Store individual MC-Dropout predictions for each iteration

    for _ in range(mc_iterations):
        predictions = []
        with torch.no_grad():
            for features, state, action, next_state in dataloader:
                inputs = torch.cat((features.float(), state.unsqueeze(1).float(), action.unsqueeze(1).float()), dim=1).to(device)
                outputs = model(inputs)
                predictions.append(outputs.cpu().numpy())
        all_mc_predictions.append(np.concatenate(predictions, axis=0))

    # Compute the average of all predictions
    avg_predictions = np.mean(all_mc_predictions, axis=0)

    return avg_predictions, all_mc_predictions


def compute_mc_dropout_metrics(model, dataloader, device, mc_iterations=10):
    """Compute metrics (log likelihood, accuracy, F1) using MC-Dropout."""
    avg_predictions, all_mc_predictions = mc_dropout_predictions(model, dataloader, device, mc_iterations)

    # Get the ground truth
    all_targets = []
    with torch.no_grad():
        for features, state, action, next_state in dataloader:
            all_targets.append(next_state.unsqueeze(1).float().cpu().numpy())

    all_targets = np.concatenate(all_targets, axis=0)
    predicted_binary = avg_predictions.round()

    # Compute metrics
    accuracy = compute_accuracy(predicted_binary, all_targets)
    log_likelihood = compute_log_likelihood(avg_predictions, all_targets)
    f1 = compute_f1_score(predicted_binary, all_targets)

    return log_likelihood, accuracy, f1, all_mc_predictions


def compute_accuracy(predicted_binary, ground_truths):
    predicted_engagement = np.array(predicted_binary)
    actual_engagement = np.array(ground_truths)
    accuracy = np.mean(predicted_engagement == actual_engagement)
    return accuracy


def compute_log_likelihood(predicted_binary, ground_truths):
    actual_engagement = np.array(ground_truths)
    predicted_probs = np.clip(predicted_binary, 1e-10, 1 - 1e-10)
    log_likelihood = np.mean(actual_engagement * np.log(predicted_probs) +
                             (1 - actual_engagement) * np.log(1 - predicted_probs))
    return log_likelihood


def compute_f1_score(predicted_binary, ground_truths):
    return f1_score(np.array(ground_truths), np.array(predicted_binary))


def logistic_growth(t, initial_mothers, L, k, t0):
    """Logistic growth model to simulate the number of new mothers joining each month."""
    return initial_mothers + (L - initial_mothers) / (1 + np.exp(-k * (t - t0)))


def main(args):
    features, state_trajectories, action_trajectories = data_preprocessing(args.data_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    initial_mothers = 100
    L = 3000
    k = 0.4
    t0 = 10
    num_months = 10
    log_likelihoods = []
    accuracies = []
    f1_scores = []  # To store F1 scores


    registered_mothers = set()
    mother_join_times = {}

    mc_iterations = 50  # Number of MC-Dropout iterations

    def aggregate_to_monthly(data, start_week, end_week):
        """Aggregate weekly data into monthly data."""
        data_array = np.array(data[start_week:end_week])
        monthly_state = 1.0 if np.any(data_array > 30) else 0.0
        return monthly_state

    for month in range(0, 2):
        num_mothers = int(logistic_growth(month, initial_mothers, L, k, t0))
        num_arms = num_mothers

        new_mothers = set(range(len(registered_mothers), num_mothers))
        registered_mothers.update(new_mothers)

        for new_mother in new_mothers:
            mother_join_times[new_mother] = month

    for month in range(2, num_months):  # Starting prediction from month 2
        ground_truths_by_month = []  # To store ground truths for each month
        num_mothers = int(logistic_growth(month, initial_mothers, L, k, t0))
        num_arms = num_mothers

        new_mothers = set(range(len(registered_mothers), num_mothers))
        registered_mothers.update(new_mothers)

        for new_mother in new_mothers:
            mother_join_times[new_mother] = month

        # Create datasets for this month
        train_data, test_data, ground_truths = [], [], []  # Collect ground truths
        for arm in range(num_arms):
            arm_features = np.array(features[arm])
            join_time = mother_join_times.get(arm, float('inf'))
            relative_month = month - join_time

            if relative_month < 0:
                continue

            if relative_month == 0:
                test_ground_truth = aggregate_to_monthly(state_trajectories[arm], 0, 4)
                test_data.append((arm_features, 1, 0, test_ground_truth))
                ground_truths.append(test_ground_truth)  # Collect ground truth

            elif relative_month == 1:
                history_start_week = 0
                history_end_week = 4
                next_start_week = 4
                next_end_week = 8
                last_month_state = aggregate_to_monthly(state_trajectories[arm], history_start_week, history_end_week)
                last_month_action = aggregate_to_monthly(action_trajectories[arm], history_start_week, history_end_week)
                test_ground_truth = aggregate_to_monthly(state_trajectories[arm], next_start_week, next_end_week)
                test_data.append((arm_features, last_month_state, last_month_action, test_ground_truth))
                ground_truths.append(test_ground_truth)  # Collect ground truth

            else:
                for past_month in range(1, relative_month):
                    history_start_week = (past_month - 1) * 4
                    history_end_week = past_month * 4
                    history_month_state = aggregate_to_monthly(state_trajectories[arm], history_start_week, history_end_week)
                    history_month_action = aggregate_to_monthly(action_trajectories[arm], history_start_week, history_end_week)
                    next_start_week = past_month * 4
                    next_end_week = min(next_start_week + 4, len(state_trajectories[arm]))
                    next_month_state = aggregate_to_monthly(state_trajectories[arm], next_start_week, next_end_week)
                    train_data.append((arm_features, history_month_state, history_month_action, next_month_state))

                last_month_state = aggregate_to_monthly(state_trajectories[arm], (relative_month - 1) * 4, relative_month * 4)
                last_month_action = aggregate_to_monthly(action_trajectories[arm], (relative_month - 1) * 4, relative_month * 4)
                test_ground_truth = aggregate_to_monthly(state_trajectories[arm], relative_month * 4, min((relative_month + 1) * 4, len(state_trajectories[arm])))
                test_data.append((arm_features, last_month_state, last_month_action, test_ground_truth))
                ground_truths.append(test_ground_truth)  # Collect ground truth

        # Save the ground truths for the month
        ground_truths_by_month.append(ground_truths)
        np.save(f"ground_truths_month_{month}.npy", np.array(ground_truths, dtype=object))  # Save ground truths

        # Split data for training and validation
        train_size = int(0.8 * len(train_data))
        val_size = len(train_data) - train_size
        train_subset, val_subset = random_split(train_data, [train_size, val_size])

        train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

        input_size = len(features[0]) + 2  # Number of features + state + action
        hidden_size = 64
        model = StatePredictor(input_size, hidden_size)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        trained_model, best_val_loss = train_model_with_early_stopping(
            model, criterion, optimizer, train_loader, val_loader, device,
            save_best=args.save_best, num_epochs=50, patience=10
        )

        test_log_likelihood, test_accuracy, test_f1_score, all_mc_predictions = compute_mc_dropout_metrics(
            trained_model, test_loader, device, mc_iterations=mc_iterations
        )

        np.save(f"mc_predictions_month_{month}.npy", np.array(all_mc_predictions))
        np.save(f"ground_truths_month_{month}.npy", np.array(ground_truths_by_month, dtype=object))

        print(f"Month {month}: Log Likelihood = {test_log_likelihood:.4f}, Accuracy = {test_accuracy:.4f}, F1 Score = {test_f1_score:.4f}")

        log_likelihoods.append(test_log_likelihood)
        accuracies.append(test_accuracy)
        f1_scores.append(test_f1_score)



    # Plot metrics
    months = list(range(2, num_months))

    # Plot Log Likelihood
    fig, ax = plt.subplots()
    ax.plot(months, log_likelihoods, 'b-', label='Log Likelihood')
    ax.set_xlabel('Month')
    ax.set_ylabel('Log Likelihood', color='b')
    ax.tick_params('y', colors='b')
    plt.title('Log Likelihood Over Months with MC-Dropout')
    plt.tight_layout()
    plt.savefig('log_likelihood_plot.png', dpi=300, bbox_inches='tight')  # Save the figure
    plt.show()

    # Plot Accuracy
    fig, ax = plt.subplots()
    ax.plot(months, accuracies, 'r--', label='Accuracy')
    ax.set_xlabel('Month')
    ax.set_ylabel('Accuracy', color='r')
    ax.tick_params('y', colors='r')
    plt.title('Accuracy Over Months with MC-Dropout')
    plt.tight_layout()
    plt.savefig('accuracy_plot.png', dpi=300, bbox_inches='tight')  # Save the figure
    plt.show()

    # Plot F1 Score
    fig, ax = plt.subplots()
    ax.plot(months, f1_scores, 'g-', label='F1 Score')
    ax.set_xlabel('Month')
    ax.set_ylabel('F1 Score', color='g')
    ax.tick_params('y', colors='g')
    plt.title('F1 Score Over Months with MC-Dropout')
    plt.tight_layout()
    plt.savefig('f1_score_plot.png', dpi=300, bbox_inches='tight')  # Save the figure
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="trajectories_continuous.json", help="Path to the data file.")
    parser.add_argument("--save_best", type=bool, default=True, help="Save the best model based on validation loss (True) or the last epoch model (False).")
    args = parser.parse_args()

    main(args)
