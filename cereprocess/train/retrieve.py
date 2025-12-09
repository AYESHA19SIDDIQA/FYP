import os
import pickle
import pandas as pd
from .callbacks import History
pd.set_option("display.max_rows", None)


def get_results(destdir, clean = False):
    ''' Get the data of all results to view them and check them
    '''
    csvpath = os.path.join(destdir, "results.csv")
    df = pd.read_csv(csvpath)
    if (clean == True):
        df = df.drop(["hyperparamters des", "model des", "data des", "history des"], axis = 1)
        df_decimal = df.select_dtypes(include=['float', 'int'])  # Select decimal columns
        df[df_decimal.columns] = df_decimal.round(4)
    return df

def get_paths(res, model_id):
    ''' Gets all the paths of history, hyperparameters, model description, data description
    '''
    hyp_path = list(res[res["Model ID"] == model_id]["hyperparamters des"])[0]
    model_path = list(res[res["Model ID"] == model_id]["model des"])[0]
    data_path = list(res[res["Model ID"] == model_id]["data des"])[0]
    history_path = list(res[res["Model ID"] == model_id]["history des"])[0]
    return hyp_path, model_path, data_path, history_path

def load_picks(res, model_id):
    ''' Loads all the stored pickles from the results
    '''
    hyp_path, model_path, data_path, history_path = get_paths(res, model_id)
    try:
        with open(hyp_path, 'rb') as file:
            hyp = pickle.load(file)
    except:
        print("Hyper Parameters were Not Loaded")
        hyp = None
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
    except:
        print("Model Description Not Loaded")
        model = None
    try:
        with open(data_path, 'rb') as file:
            data = pickle.load(file)
    except:
        print("Data Description Not Loaded")
        data = None
    try:
        with open(history_path, 'rb') as file:
            history = pickle.load(file)
    except:
        print("History Not Loaded")
        history = None
    return hyp, model, data, history
