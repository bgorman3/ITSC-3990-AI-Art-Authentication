import torch
import os
import csv

def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)

def load_model(model, filepath):
    model.load_state_dict(torch.load(filepath))
    return model

def ensure_dir_exists(dir_path):
    """
    Ensure that a directory exists. If it does not exist, create it.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def save_to_csv(data, filepath, headers):
    """
    Save data to a CSV file.
    """
    ensure_dir_exists(os.path.dirname(filepath))
    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(data)

    