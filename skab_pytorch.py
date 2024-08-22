#!/usr/bin/env python3
import json
import logging
import os
import pickle
import sys
from typing import Iterable, Optional, NamedTuple, cast

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import f1_score
import torch
from codetiming import Timer
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torcheval.metrics.functional import binary_f1_score

# noinspection SpellCheckingInspection
DATA_PATH = '../kaggle/input/ml-cource-cifrum-anomaly-public'
MODEL_FILENAME = 'model/skab.pt'
SCALE_FILENAME = 'model/skab_scale.json'
OPTUNA_LOG_FILENAME = 'optuna_log.csv'
MODE = 'use'  # tune / train / use

WINDOW_LENGTH = 221
BATCH_SIZE = 256  # 8
LAYERS = 5  # hidden GRU layers
HIDDEN_SIZE = 31  # the size of each hidden layer
DROPOUT = 0.027043278483917  # GRU dropout
LEARNING_RATE = 0.000668650863057288
EPOCHS = 1  # 19
SHUFFLE = True  # reshuffle the data at every epoch
OPTIMIZER = torch.optim.Adam
torch.manual_seed(0)

# TODO: EarlyStopping

# TODO: Debug 0.0 results


def to_df(tensor: Tensor):
    return pd.DataFrame(tensor.detach().cpu().numpy())


def mute_log(message_substring: str, logger: str = 'root') -> None:
    """ Mute log messages containing a substring. """
    logging.getLogger(logger).addFilter(lambda record: message_substring not in record.msg)


def iterate_dataframes(path: str, anomaly_free: bool = True) -> Iterable[pd.DataFrame]:
    """ Find and read dataframes at the specified path. """
    path = os.path.join(DATA_PATH, path)

    # CSV
    if path.endswith('.csv'):
        yield pd.read_csv(path, sep=';', index_col='datetime', parse_dates=True)
        return

    # Directory with CSVs
    if os.path.isdir(path):
        paths = (os.path.join(root, file)
                 for root, _, files in os.walk(path)
                 for file in files if file.endswith('.csv') and (anomaly_free or 'anomaly-free' not in file))
        for csv_path in paths:
            yield pd.read_csv(csv_path, sep=';', index_col='datetime', parse_dates=True)
        return

    # Pickle file
    with open(path, 'rb') as file:
        data = pickle.load(file)
        if isinstance(data, Iterable):
            yield from data
        else:
            yield data


class Scale(NamedTuple):
    """ Feature scales """
    in_std: pd.Series  # sensor data standard deviation
    in_mean: pd.Series  # sensor data mean
    out_std: pd.Series  # generated feature standard deviation
    out_mean: pd.Series  # generated feature mean


class Data(NamedTuple):
    """ Engineered data """
    data: pd.DataFrame  # sensor data
    labels: pd.DataFrame  # anomalies, change points
    scale: Scale  # data standard deviation and mean


def feature_engineering(data: pd.DataFrame, sort: bool, scale: Optional[Scale] = None) -> Data:
    # Get sensor data and anomaly/changepoint labels.
    data = data.fillna(0.0)
    if sort:
        if 'anomaly' in data:
            data = data.sort_values(['anomaly', 'changepoint'], ascending=False)
        data = data[~data.index.duplicated()].sort_index()
    if 'anomaly' in data:
        labels = data[['anomaly', 'changepoint']]
        data = data[data.columns.difference(labels.columns)]
    else:
        labels = pd.DataFrame(0.0, index=data.index, columns=['anomaly', 'changepoint'])

    # Scale raw data.
    if scale:
        in_std = scale.in_std
        in_mean = scale.in_mean
    else:
        in_std = data.std()
        in_mean = data.mean()
    data = (data - in_mean) / in_std

    # Generate new features.
    data['Time step'] = data.index.diff().seconds.to_series(index=data.index).apply(np.log).fillna(0.0)
    data['Acceleration mean'] = data['Accelerometer1RMS'] + data['Accelerometer2RMS']
    data['Acceleration diff'] = data['Accelerometer1RMS'] - data['Accelerometer2RMS']
    data['Temperature mean'] = data['Temperature'] + data['Thermocouple']
    data['Temperature diff'] = data['Temperature'] - data['Thermocouple']
    data = data.drop(['Accelerometer1RMS', 'Accelerometer2RMS', 'Temperature', 'Thermocouple'], axis=1)

    # Scale new features.
    if scale:
        out_std = scale.out_std
        out_mean = scale.out_mean
    else:
        out_std = data.std()
        out_mean = data.mean()
    data = (data - out_mean) / out_std

    return Data(data, labels, Scale(in_std, in_mean, out_std, out_mean))


class WindowDataset(torch.utils.data.Dataset[tuple[torch.Tensor, ...]]):
    """ A time series dataset with a sliding window getter """
    def __init__(self, *tensors: Tensor, window: int):
        assert tensors
        self.tensors = tensors
        self.window = window

    def __len__(self) -> int:
        return self.tensors[0].size(0) - self.window + 1

    def __getitem__(self, index):
        return tuple(tensor[index:index+self.window] for tensor in self.tensors)


def get_dataloader(data: Data,
                   batch_size: int = 1,
                   window: Optional[int] = None,
                   shuffle: bool = False,
                   device: Optional[str | torch.device | int] = None) -> DataLoader[tuple[Tensor, ...]]:
    data_tensor = torch.tensor(data.data.values, dtype=torch.float32, device=device)
    label_tensor = torch.tensor(data.labels['anomaly'].values, dtype=torch.float32, device=device)
    window = window or data_tensor.size(0)
    dataset = WindowDataset(data_tensor, label_tensor, window=window)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class TimeSeriesBinaryClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        # input: input_size features
        # hidden: hidden_size * (num_layers for GRU + 1 linear)
        # output: anomaly probability
        self.gru = nn.GRU(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        gru_output, _ = self.gru(x)
        logits = self.linear(gru_output)
        prob = self.sigmoid(logits)
        return prob.squeeze()


def train(window_length: int,
          batch_size: int,
          layers: int,
          hidden_size: int,
          dropout: float,
          learning_rate: float,
          epochs: int) -> float:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device}.")
    mute_log("Some classes do not exist in the target")
    model = TimeSeriesBinaryClassifier(9, hidden_size, layers, dropout).to(device)

    train_df = pd.concat(iterate_dataframes('data', anomaly_free=True))
    train_data = feature_engineering(train_df, sort=True)
    train_dataloader = get_dataloader(train_data, batch_size, window_length, SHUFFLE, device)
    loss_fn = nn.BCELoss()
    optimizer = OPTIMIZER(model.parameters(), learning_rate)
    f1 = 0.0
    for t in range(epochs):
        print(f"Epoch {t + 1}")
        print("-------------------------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        loss, f1 = test(train_dataloader, model, loss_fn)
        print(f"Epoch loss: {loss:>4f}  F1 score: {f1:>4f}\n")

    torch.save(model.state_dict(), MODEL_FILENAME)
    with open(SCALE_FILENAME, 'w') as f:
        json.dump({name: series.to_dict() for name, series in train_data.scale._asdict().items()}, f, indent=2)
    print(f"Model saved to '{MODEL_FILENAME}'.")
    return f1


def train_loop(dataloader: DataLoader[tuple[Tensor, ...]],
               model: nn.Module,
               loss_fn: nn.Module,
               optimizer: torch.optim.Optimizer):
    total_length = len(cast(WindowDataset, dataloader.dataset))
    loss_sum = 0
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        # forward run
        pred = model(X)
        loss = loss_fn(pred, y)

        # backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # status
        loss_sum += loss.item()
        if batch % 100 == 0:
            f1 = binary_f1_score(pred[-1].round(), y[-1]).item()
            current = batch * dataloader.batch_size
            print(f"Loss: {loss_sum / 100:>3f}  F1 score: {f1:>3f}  [{current:>5d}/{total_length:>5d}]", flush=True)
            loss_sum = 0


def test(dataloader: DataLoader[tuple[Tensor, ...]], model: nn.Module, loss_fn: nn.Module) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        x, y = cast(WindowDataset, dataloader.dataset).tensors
        x = x[WINDOW_LENGTH:]
        y = y[WINDOW_LENGTH:]
        pred = model(x)
        loss = loss_fn(pred, y).item()
        f1 = binary_f1_score(pred.round(), y).item()
    return loss, f1


def test_reference(layers: int, hidden_size: int, dropout: float) -> float:
    pred = predict(layers, hidden_size, dropout)
    reference = pd.read_csv(os.path.join(DATA_PATH, 'reference.csv'), sep=';', index_col='ID')
    f1 = f1_score(reference['anomaly'], pred)
    print(f'Model F1 score: {f1}')
    return f1


def predict(layers: int = LAYERS,
            hidden_size: int = HIDDEN_SIZE,
            dropout: float = DROPOUT) -> pd.Series:
    return predict_df(pd.concat(iterate_dataframes('test.pkl')), layers, hidden_size, dropout)


def predict_df(test_df: pd.DataFrame,
               layers: int = LAYERS,
               hidden_size: int = HIDDEN_SIZE,
               dropout: float = DROPOUT) -> pd.Series:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device}.")
    mute_log("Some classes do not exist in the target")
    model = TimeSeriesBinaryClassifier(9, hidden_size, layers, dropout).to(device)

    with open(SCALE_FILENAME, 'r') as f:
        scale = Scale(**json.load(f))
    model.load_state_dict(torch.load(MODEL_FILENAME, weights_only=True))

    test_data = feature_engineering(test_df, sort=False, scale=scale)
    test_dataloader = get_dataloader(test_data, shuffle=False, device=device)

    x, _ = cast(WindowDataset, test_dataloader.dataset).tensors
    model.eval()
    with torch.no_grad():
        pred = model(x)
        anomalies = pd.Series(pred.cpu().numpy()).round().astype(int)
    return anomalies


# noinspection PyTypeChecker
@Timer(text="[{:.2f}s] ", logger=sys.stderr.write)
def objective(trial: optuna.trial.Trial):
    try:
        hyperparameters = (trial.suggest_int('window_length', 160, 360, log=True),
                           2 ** trial.suggest_int('batch_size_exp', 3, 7),
                           trial.suggest_int('layers', 4, 5),
                           trial.suggest_int('hidden_size', 8, 43, log=True),
                           trial.suggest_float('dropout', 0.007, 0.1, log=True),
                           trial.suggest_float('learning_rate', 0.00025, 0.005, log=True),
                           trial.suggest_int('epochs', 10, 40, log=True))
        print(f'Starting trial {trial.number} with parameters: {trial.params}')
        with open(OPTUNA_LOG_FILENAME, "a") as log:
            log.write(f"{trial.datetime_start:%Y-%m-%d %H:%M:%S};{';'.join(map(str, trial.params.values()))}")
        train(*hyperparameters)
        f1 = test_reference(trial.params['layers'], trial.params['hidden_size'], trial.params['dropout'])
    except BaseException as e:
        print(e)
        f1 = 0.0
    with open(OPTUNA_LOG_FILENAME, "a") as log:
        log.write(f";{f1}\n")
    return f1


@Timer(text="Total time: {:.2f} s.")
def main():
    if MODE == 'tune':
        if not os.path.isfile(OPTUNA_LOG_FILENAME):
            with open(OPTUNA_LOG_FILENAME, "w") as log:
                log.write(f"datetime;window_length;batch_size_exp;layers;hidden_size;dropout;learning_rate;epochs;F1\n")
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=0))
        study.optimize(objective, n_trials=1000)
    elif MODE == 'train':
        train(WINDOW_LENGTH, BATCH_SIZE, LAYERS, HIDDEN_SIZE, DROPOUT, LEARNING_RATE, EPOCHS)
    elif MODE == 'use':
        anomalies = predict(LAYERS, HIDDEN_SIZE, DROPOUT)
        submission = pd.DataFrame(anomalies).reset_index()
        submission.columns = ['ID', 'anomaly']
        submission.to_csv('submission.csv', index=False)
    else:
        raise Exception(f"MODE = '{MODE}'")


if __name__ == '__main__':
    main()
