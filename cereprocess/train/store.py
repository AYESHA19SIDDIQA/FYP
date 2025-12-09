import os
import pickle
import pandas as pd
import torch.nn as nn
import torch
from .callbacks import History


# We will start by saving the model
def save_model(model, path):
    '''
    Save the model to the given path
    '''
    torch.save(model.state_dict(), path)

def save_model_details(model_description, data_description, hyperparameters, history, path, model_save_name):
    '''
    Save the model description to the given path
    '''
    os.makedirs(os.path.join(path, 'model_info'), exist_ok=True)
    os.makedirs(os.path.join(path, 'data_info'), exist_ok=True)
    os.makedirs(os.path.join(path, 'hparam'), exist_ok=True)
    os.makedirs(os.path.join(path, 'history'), exist_ok=True)
    model_path = os.path.join(path, 'model_info', model_save_name + '.pkl')
    data_path = os.path.join(path, 'data_info', model_save_name  + '.pkl')
    hyp_path = os.path.join(path, 'hparam', model_save_name + '.pkl')
    history_path = os.path.join(path, 'history', model_save_name + '.pkl')
    with open(model_path, 'wb') as file:
        pickle.dump(model_description, file)
    with open(data_path, 'wb') as file:
        pickle.dump(data_description, file)
    with open(hyp_path, 'wb') as file:
        pickle.dump(hyperparameters, file)
    with open(history_path, 'wb') as file:
        pickle.dump(history, file)
    return model_path, data_path, hyp_path, history_path




def df_entry(model_description, data_description, history, hyperparameters, path, model_save_name):
    '''
    Return a pd series with the data to be stored
    '''
    model_path, data_path, hyp_path, history_path = save_model_details(model_description, data_description, hyperparameters, history, path, model_save_name)
    res = {
        'Model Name': model_description['name'],
        'Model Size': model_description['size'],
        'Model ID': model_description['id'],
        'Data ID': data_description['id'],
        'hyperparamters des': hyp_path,
        'model des': model_path,
        'data des': data_path,
        'history des': history_path
        }
    for div in ['train', 'val']:
        for key, value in history.history[div].items():
            res[div + "_" + key] = value[-1]

    res["Best Val Accuracy"] = history.best["accuracy"]["accuracy"]
    res["Best Val Loss"] = history.best["loss"]["loss"]

    return pd.Series(res)

def update_csv(path, model_description, data_description, history, hyperparameters, model_save_name):
    '''
    Update the csv file with the new data
    If the csv file does not exist, it creates a new one
    '''
    csvpath = os.path.join(path, "results.csv")
    try:
        df = pd.read_csv(csvpath)
    except FileNotFoundError:
        df = pd.DataFrame()
    data = df_entry(model_description, data_description, history, hyperparameters, path, model_save_name)
    df = pd.concat([df, data.to_frame().T], ignore_index=True)
    df.to_csv(csvpath, index=False)
