import json
import numpy as np

def data_preprocessing(data_path, groups=['control']):
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


def map_features_to_prompt(features, past_listening_times, past_actions):
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
        try:
            return category_list[slice.index(1)]
        except ValueError:
            return "Unknown"

    # Modify past behavior to show "Engaged" or "Not Engaged" based on past listening times
    past_behavior = "\n".join(
        [f"  - Week {i+1}: {'Engaged' if time > 30 else 'Not Engaged'}, Action: {'Received service call' if action == 1 else 'No service call'}"
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

    return mapped_features