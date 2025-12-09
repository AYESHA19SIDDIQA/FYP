import os
import torch
from .train import train
from ..datasets.dataset import Dataset
from torch.utils.data import DataLoader
from ..datasets.pytordataset import KFoldDataset, EEGDataset
from .misc import get_model_size
from .store import update_csv
import numpy as np
import pandas as pd

# def evaluate(model, val_loader, criterion, device, metrics, history):

def _get_modelsummary(model, input_size):
    model_summary = model.__str__()
    return model_summary

def _get_datasummary(datapath, pipeline):
    #TODO
    return ""

def _calc_inputsize(s_rate, t_span, c_no):
    return (c_no, s_rate * t_span)

def get_dataloaders(datadir, batch_size, seed, target_length=None, indexes=None):
    """
    Utility to get train and eval DataLoaders with optional time-step cropping.

    Args:
        datadir (string): Root directory containing 'train' and 'eval' subdirs.
        batch_size (int): Batch size for loaders.
        seed (int): Random seed for shuffling.
        target_length (int, optional): Number of time steps to crop each sample to.
        indexes (list of int, optional): Subset of dataset indices to include.

    Returns:
        train_loader, eval_loader
    """
    train_dir = os.path.join(datadir, 'train')
    eval_dir = os.path.join(datadir, 'eval')

    trainset = EEGDataset(train_dir, indexes=indexes, target_length=target_length)
    evalset = EEGDataset(eval_dir,  indexes=indexes, target_length=target_length)

    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=lambda _: np.random.seed(seed)
    )
    eval_loader = DataLoader(
        evalset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, eval_loader

def get_dataloaders_restricted(datadir, batch_size, seed, target_length=None, indexes=None, limit_normal=500):
    train_dir = os.path.join(datadir, 'train')
    eval_dir = os.path.join(datadir, 'eval')

    # Load train CSV
    annotations_file = os.path.join(train_dir, 'data.csv')
    df = pd.read_csv(annotations_file)

    # Filter normal and abnormal
    normal_df = df[df['Label'] == 0]
    abnormal_df = df[df['Label'] == 1]
    print("Abnormal",len(abnormal_df))
    # Randomly sample 500 normal indices
    np.random.seed(seed)
    sampled_normal = normal_df.sample(n=limit_normal, replace=False, random_state=seed)
    print("Normal",len(sampled_normal))
    # Combine sampled normal + all abnormal
    final_df = pd.concat([sampled_normal, abnormal_df]).reset_index(drop=True)

    # Get corresponding indices in the original CSV (if needed)
    selected_indices = final_df.index.tolist()
    print("selected_inex:",len(selected_indices))

    # Create datasets
    trainset = EEGDataset(train_dir, indexes=selected_indices, target_length=target_length)
    evalset = EEGDataset(eval_dir, indexes=indexes, target_length=target_length)  # full eval set

    # Create loaders
    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=lambda _: np.random.seed(seed)
    )
    eval_loader = DataLoader(
        evalset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, eval_loader



def _write_counter(counter, destpath, filename='counter.txt'):
    filename = os.path.join(destpath, filename)
    with open(filename, 'w') as file:
        file.write(str(counter))

def _read_counter(destpath, filename='counter.txt'):
    filename = os.path.join(destpath, filename)
    if not os.path.exists(filename):
        return 0  # Return a default value (e.g., 0) if the file does not exist

    with open(filename, 'r') as file:
        content = file.read()
        return int(content)  # or float(content) if you expect a float

def _increment_counter(destpath, filename='counter.txt'):
    filename = os.path.join(destpath, filename)
    counter = _read_counter(destpath)
    print("Incremented")
    counter += 1
    _write_counter(counter, destpath)

def get_datanpz(datapath, destdir, pipeline, input_size):
    # Making the dataset from its pipeline
    dataset = Dataset(datapath)
    dataset.set_pipeline(pipeline)
    datadir = os.path.join(destdir, 'data')
    print(dataset.save_all(datadir))
    datadir, s_rate, t_span, c_no, data_id = dataset.save_all(datadir)
    if (input_size != _calc_inputsize(s_rate, t_span, c_no)):
        print("Input Size given: ", input_size)
        print("Calculated: ", _calc_inputsize(s_rate, t_span, c_no))
        raise ValueError("Input Size Mismatch")

    data_description = {}
    datasummary = _get_datasummary(datapath, pipeline)

    # Adding the data description
    data_description['id'] = data_id
    data_description['pipeline'] = pipeline.__class__.__name__
    data_description['sampling_rate'] = s_rate
    data_description['time_span'] = t_span
    data_description['channel_no'] = c_no
    data_description['summary'] = datasummary

    return datadir, data_description


def oneloop(device, model, train_loader, eval_loader, data_description, hyperparameters, trainelements, destdir, model_name = None, save_best_acc = False, scheduler=None):
    input_size = _calc_inputsize(data_description["sampling_rate"], data_description["time_span"], data_description["channel_no"])
    # We will start by initializing the model and data description
    model_description = {}
    if model_name is None:
        model_description['name'] = model.__class__.__name__
    else:
        model_description['name'] = model_name
    model_description["size"] = str(get_model_size(model))

    # Resetting cuda for every loop
    if device == 'cuda':
        torch.cuda.empty_cache()

    # Extracting the necessary elements
    metrics = trainelements.metrics
    history = trainelements.history
    criterion = trainelements.criterion
    earlystopping = trainelements.earlystopping

    # Generating the model and data summaries
    modelsummary = _get_modelsummary(model, input_size)



    # Adding the model description
    model_description['summary'] = modelsummary
    model_description['id'] = model.__class__.__name__ + '_' + data_description["id"]

    optimizer = trainelements.optimizer(model.parameters(), lr=hyperparameters['lr'])
    if scheduler:
        sch = scheduler(optimizer, T_max=1000)
    else:
        sch = None

    # Finally training everything
    model_save_dir = os.path.join(destdir, 'models')
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_name = "model_" + str(_read_counter(destdir))
    _increment_counter(destdir)

    model_description['id'] = model_save_name

    model_save_path = os.path.join(model_save_dir, model_save_name + '.pt')
    earlystopping.path = model_save_path
    train(model, train_loader, eval_loader, optimizer, criterion, hyperparameters['epochs'], history, metrics, device, model_save_path, earlystopping, accum_iter=hyperparameters['accum_iter'], save_best_acc=save_best_acc, scheduler=sch)
    update_csv(destdir, model_description, data_description, history, hyperparameters, model_save_name)
    return model











