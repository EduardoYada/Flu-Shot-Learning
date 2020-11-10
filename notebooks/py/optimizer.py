
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os
from tqdm import trange
import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision import transforms
from net import DEVICE
import optuna


# DEVICE = torch.device("cpu")
BATCHSIZE = 128
CLASSES = 1 
DIR = os.getcwd()
TARGET = 'h1n1_vaccine'
EPOCHS = 10
LOG_INTERVAL = 10
N_TRAIN_EXAMPLES = BATCHSIZE * 30
N_VALID_EXAMPLES = BATCHSIZE * 10



class OptunaData(Dataset):

    def __init__(self, x, y):
        
        scl = MinMaxScaler()

        for col in x:
            # parsing as numeric data 
            x[col] = x[col].astype('category').cat.codes


        # XXX: Normalization
        x = scl.fit_transform(x.values)
        n_features = x.shape[1]
        n_samples = x.shape[0]

        self.x =  torch.tensor(x, dtype=torch.float32).to(DEVICE)
        self.y = torch.tensor(y.values, dtype=torch.float32).to(DEVICE)
        self.n_features = n_features
        self.n_samples = n_samples

    def __getitem__(self, index):
       return  self.x[index], self.y[index]
    
    def __len__(self):
        return self.n_samples



def define_model(trial):
    # We optimize the number of layers, hidden units and dropout ratio in each layer.
    n_layers = trial.suggest_int("n_layers", 1, 6)
    layers = []

    #constatns for our dataset
    in_features = 36 
    for i in range(n_layers):
        out_features = trial.suggest_int("n_units_l{}".format(i), 4, 128)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        p = trial.suggest_float("dropout_l{}".format(i), 0.2, 0.5)
        layers.append(nn.Dropout(p))

        in_features = out_features
    layers.append(nn.Linear(in_features, CLASSES))

    return nn.Sequential(*layers)


def get_mnist():

    data = pd.read_csv('../../data/training_set_features.csv')
    data['target'] = pd.read_csv('../../data/training_set_labels.csv')[TARGET]

    train, val  = train_test_split(data)

    train_data, val_data = OptunaData(train.drop[TARGET], train[TARGET]),  \
                           OptunaData(val.drop[TARGET], val[TARGET])
    print(train_data, val_data)
    train_loader, valid_loader = DataLoader(train_data, batch_size=BATCHSIZE) , \
                                DataLoader(val_data, batch_size=BATCHSIZE)
     
    return train_loader, valid_loader


def objective(trial):

    # Generate the model.
    model = define_model(trial).to(DEVICE)

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # Get the MNIST dataset.
    train_loader, valid_loader = get_mnist()

    # Training of the model.
    for epoch in range(EPOCHS):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # Limiting training data for faster epochs.
            if batch_idx * BATCHSIZE >= N_TRAIN_EXAMPLES:
                break

            optimizer.zero_grad()
            output = model(data)
            loss = torch.nn.BCELoss(output, target.reshape(-1, 1))
            loss.backward()
            optimizer.step()

        # Validation of the model.
        model.eval()
        correct = 0
        for batch_idx, (data, target) in enumerate(valid_loader):
            # Limiting validation data.
            if batch_idx * BATCHSIZE >= N_VALID_EXAMPLES:
                break
            data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)
            output = model(data)
            # Get the index of the max log-probability.
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = correct / min(len(valid_loader.dataset), N_VALID_EXAMPLES)

        trial.report(accuracy, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, timeout=600)

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))