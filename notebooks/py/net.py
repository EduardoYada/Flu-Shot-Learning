import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset , DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data.dataset import random_split # cv for torch
from tqdm import trange
import os.path
import random
# XXX: 

np.random.seed(1)
random.seed()

# XXX: t.set_description( f'')
TARGET =  'h1n1_vaccine'
EPOCHS = 1000
LEARNING_RATE = 0.01
BATCH_SIZE = 128
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class Net(nn.Module):

    def __init__(self, input):
        super(Net, self).__init__()
        self.int = nn.Linear(input, 64)
        self.l2 = nn.Linear(64, 128)
        self.l3  = nn.Linear(128, 64)
        self.l4 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 1)

    def forward(self, x):
        # x = nn.functional.relu(self.int(x))
        x = self.int(x) 
        x = torch.relu(self.l2(x))
        x = torch.relu(self.l3(x))
        x = self.l4(x)
        x = torch.sigmoid(self.out(x))
        return x

class IngestData(Dataset):

    def __init__(self, train_path='../../data/training_set_features.csv',
                       test_path='../../data/training_set_labels.csv'):
        
        x = pd.read_csv(train_path)
        n_features = x.shape[1]
        y = pd.read_csv(test_path)['h1n1_vaccine']
        scl = MinMaxScaler()

        for col in x:
            # parsing as numeric data 
            x[col] = x[col].astype('category').cat.codes


        # XXX: Normalization
        x = scl.fit_transform(x.values)

        self.x = torch.tensor(x, dtype=torch.float32).to(DEVICE)
        self.y = torch.tensor(y.values, dtype=torch.float32).to(DEVICE)
        self.n_samples = x.shape[0]
        self.n_features = n_features

    def __getitem__(self, index):

        return self.x[index], self.y[index] 

    def __len__(self):

        return self.n_samples

def train_my_model(model, data, loss_fn=torch.nn.BCELoss(),  optimizer=None, epochs=EPOCHS):
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoc in (t := trange(epochs)):

        model.train()
        
        for x_batch, y_batch in data:
             # Makes predictions
            yhat = model(x_batch)
             # Computes loss
            loss = loss_fn(yhat, y_batch.reshape(-1, 1))
             # Computes gradients
            loss.backward()
             # Updates parameters and zeroes gradients
            optimizer.step()
            optimizer.zero_grad()

        t.set_description(f'EPOCH {epoc + 1} | loss: {loss.item(): .4f}')

        # t.set_description(f'EPOCH {epoc + 1} | loss: {len(y_batch): .4f}')


def main():
    
    data = IngestData()
    n_features = data.n_features
    print(n_features)
    data = DataLoader(data, batch_size=64, shuffle=False)
    model = Net(n_features)

    train_my_model(model, data)

if __name__ == '__main__':
    main()