import torch  # type: ignore
import os
import csv

def save_model(model, filepath):
    """
    Save the state dictionary of a PyTorch model to a specified file path.

    Parameters:
    - model (torch.nn.Module): The model to save.
    - filepath (str): The file path to save the model state dictionary.
    """
    torch.save(model.state_dict(), filepath)

def load_model(model, filepath):
    """
    Load the state dictionary from a specified file path into a PyTorch model.

    Parameters:
    - model (torch.nn.Module): The model to load the state dictionary into.
    - filepath (str): The file path to load the model state dictionary from.

    Returns:
    - model (torch.nn.Module): The model with the loaded state dictionary.
    """
    model.load_state_dict(torch.load(filepath))
    return model

def ensure_dir_exists(dir_path):
    """
    Ensure that a directory exists. If it does not exist, create it.

    Parameters:
    - dir_path (str): The path of the directory to check/create.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def save_to_csv(data, filepath, headers):
    """
    Save data to a CSV file with specified headers.

    Parameters:
    - data (list of list): The data to save to the CSV file.
    - filepath (str): The file path to save the CSV file.
    - headers (list of str): The headers for the CSV file.
    """
    ensure_dir_exists(os.path.dirname(filepath))
    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(data)