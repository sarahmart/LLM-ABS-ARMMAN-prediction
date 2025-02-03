import json
import numpy as np

from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def data_preprocessing(data_path, groups=['random']):
    """Preprocess the data from the input JSON file for the groups specified."""

    with open(data_path, 'r') as f:
        data = json.load(f)

    for group in groups:
        group_actions = np.array(data[group]['actions'])
        print(f'Group {group}: Avg pulls = {np.average(group_actions)}') # mean engagement over *all* calls to all mothers

    num_arms = 3000  # Total number of arms (participants)
    num_steps = 39  # Number of time steps
    # print(f'Number of steps: {num_steps}')
    
    # Initialize empty trajectoriess
    state_trajectories = [[] for _ in range(num_arms)]
    action_trajectories = [[] for _ in range(num_arms)]
    features = []

    # Populate trajectories
    i = 0
    for group in groups:
        group_actions = np.array(data[group]['actions'])
        group_states = np.array(data[group]['states'])
        group_features = data[group]['features']
        features.extend(group_features)
        for arm in range(3000):
            for j in range(num_steps):
                state_trajectories[i].append(group_states[arm, j])
                action_trajectories[i].append(group_actions[arm, j])
            state_trajectories[i].append(group_states[arm, num_steps])
            i += 1
        ## each state trajectory is of length 40
        ## each action trajectory is of length 39
    
    return features, state_trajectories, action_trajectories


def map_features_to_prompt(features, past_listening_times, past_actions, first_week=False, no_actions=False):
    """Map features to their corresponding categories and return a dictionary."""
    age_categories = ["<20", "20-24", "25-29", "30-34", "35+"]
    languages = ["Hindi", "Marathi", "Kannada", "Gujarati", "English"]
    education_levels = ["1 - 5", "6 - 9", "10 Pass", "12 Pass", "Graduate", "Post Graduate", "Illiterate"]
    phone_ownership = ["woman", "husband", "family"]
    call_slots = ["8:30 AM - 10:30AM", "10:30 AM - 12:30PM", "12:30 PM - 3:30PM", "3:30 PM - 5:30PM", 
                  "5:30 PM - 7:30PM", "7:30 PM - 9:30PM"]
    channel_types = ["community", "hospital", "ARMMAN"]
    income_brackets = ["0-5000", "5001-10000", "10001-15000", "15001-20000", "20001-25000", "25001-30000", "30000+"]

    def safe_index(slice, category_list):
        # Check if the slice has elements and contains at least one 1
        if len(slice) > 0 and 1 in slice:
            index = slice.index(1)  # Find the first occurrence of 1
            if index < len(category_list):  # Ensure the index exists in the category_list
                return category_list[index]
            else:
                return "Unknown"  # Index is out of bounds for the category_list
        else:
            return "Unknown"  # Slice is empty or does not contain a 1


    if not first_week:

        # Modify past behavior to show "Engaged" or "Not Engaged" based on past listening times
        # Include action details only if no_actions is False, else remove
        if no_actions:
            past_behavior = "\n".join(
                [f"  - Month {i+1}: {'Engaged' if time > 30 else 'Not Engaged'}"
                for i, time in enumerate(past_listening_times)]
            )
        else:
            past_behavior = "\n".join(
                [f"  - Month {i+1}: {'Engaged' if time > 30 else 'Not Engaged'}, Action: {'Received service call' if action == 1 else 'No service call'}"
                for i, (time, action) in enumerate(zip(past_listening_times, past_actions))]
            )
        mapped_features = {
        "enroll_gest_age": features[0],
        "enroll_delivery_status": "pregnant" if features[1] == 0 else "delivered",
        "g": features[2],
        "p": features[3],
        "s": features[4],
        "l": features[5],
        "days_to_first_call": features[6],
        "age_category": safe_index(features[7:12], age_categories),
        "language": safe_index(features[12:16], languages),
        "education_level": safe_index(features[16:23], education_levels),
        "phone_owner": safe_index(features[23:26], phone_ownership),
        "call_slot_preference": safe_index(features[26:32], call_slots),
        "channel_type": safe_index(features[32:35], channel_types),
        "income_bracket": safe_index(features[35:], income_brackets),
        "past_behavior": past_behavior,
    }

    else:
        mapped_features = {
        "enroll_gest_age": features[0],
        "enroll_delivery_status": "pregnant" if features[1] == 0 else "delivered",
        "g": features[2],
        "p": features[3],
        "s": features[4],
        "l": features[5],
        "days_to_first_call": features[6],
        "age_category": safe_index(features[7:12], age_categories),
        "language": safe_index(features[12:16], languages),
        "education_level": safe_index(features[16:23], education_levels),
        "phone_owner": safe_index(features[23:26], phone_ownership),
        "call_slot_preference": safe_index(features[26:32], call_slots),
        "channel_type": safe_index(features[32:35], channel_types),
        "income_bracket": safe_index(features[35:], income_brackets),
    }

    return mapped_features


def scale_features(features_array):
    """Scale only features that have larger ranges."""

    scaled_features = features_array.copy()
    
    # Scale enroll_gest_age (first column)
    gest_age = features_array[:, 0]
    scaled_features[:, 0] = (gest_age - np.mean(gest_age)) / np.std(gest_age)
    
    # Scale days_to_first_call (column 6)
    days = features_array[:, 6]
    scaled_features[:, 6] = (days - np.mean(days)) / np.std(days)
    
    # other cols not changed
    return scaled_features


def select_representative_mothers(features_array, state_trajectories, n_clusters, random_state=42):
    """
    Select representative mothers using full engagement patterns and demographic features.
    """
    
    scaled_features = scale_features(features_array)
    
    # engagement pattern features for each mother
    engagement_features = []
    for trajectory in state_trajectories:
        pattern = {
            'mean_engagement': np.mean(trajectory),
            'engagement_variance': np.var(trajectory),
            'trend': np.polyfit(range(len(trajectory)), trajectory, 1)[0], # linear trend of engagement
        }
        engagement_features.append(list(pattern.values()))
    
    engagement_features = np.array(engagement_features)
    
    # scale 
    scaler = StandardScaler()
    scaled_engagement = scaler.fit_transform(engagement_features)
    
    # demographic + engagement features
    combined_features = np.hstack([scaled_features, scaled_engagement])
    
    # k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(combined_features)
    
    # For each cluster, select mothers with similar patterns
    selected_indices = []
    for cluster in range(n_clusters):
        cluster_mask = cluster_labels == cluster
        cluster_mothers = np.where(cluster_mask)[0]
        
        if len(cluster_mothers) > 0:
            # Select mother closest to cluster center
            cluster_center = kmeans.cluster_centers_[cluster]
            distances = cdist(combined_features[cluster_mothers], [cluster_center])
            closest_mother_idx = cluster_mothers[np.argmin(distances)]
            selected_indices.append(closest_mother_idx)
    
    print("\nSelected mothers engagement stats:")
    print("Full trajectory statistics:")
    
    selected_means = [np.mean(state_trajectories[i]) for i in selected_indices]
    selected_vars = [np.var(state_trajectories[i]) for i in selected_indices]
    selected_trends = [np.polyfit(range(len(state_trajectories[i])), state_trajectories[i], 1)[0] 
                      for i in selected_indices]
    
    print(f"Mean engagement: {np.mean(selected_means):.3f} (±{np.std(selected_means):.3f})")
    print(f"Mean variance: {np.mean(selected_vars):.3f} (±{np.std(selected_vars):.3f})")
    print(f"Mean trend: {np.mean(selected_trends):.3f} (±{np.std(selected_trends):.3f})")
    
    return selected_indices