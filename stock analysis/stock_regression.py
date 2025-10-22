
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle


class Stock_Model(torch.nn.Module):
    
    def __init__(self, input_size, output_size=1, hidden_layer_size=512):
        super(Stock_Model, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_layer_size)

        self.activation1 = nn.ReLU()

        self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size)

        self.activation2 = nn.ReLU()

        self.fc3 = nn.Linear(hidden_layer_size, output_size)

        with open('scaler_X.pkl', 'rb') as f:
            self.scaler_X = pickle.load(f)
        with open('scaler_y.pkl', 'rb') as f:
            self.scaler_y = pickle.load(f)

    def forward(self, X):
        output = self.fc1(X)
        output = self.activation1(output)
        output = self.fc2(output)
        output = self.activation2(output)
        output = self.fc3(output)

        return output
    
    def evaluation(self, val_loader, loss_function):
        self.eval()
        val_loss = 0.0
        predictions = []
        y_true = []

        with torch.no_grad():

            for X_val, y_val in val_loader:
                outputs = self(X_val)
                loss = loss_function(outputs, y_val)
                val_loss += loss.item() * X_val.size(0)

                predictions.append(outputs.numpy())
                y_true.append(y_val.numpy())

            predictions = np.vstack(predictions)
            y_true = np.vstack(y_true)
            
            predictions = self.scaler_y.inverse_transform(predictions)
            y_true = self.scaler_y.inverse_transform(y_true)
            
            #normalization
            val_loss /= len(val_loader.dataset)

            percent_error = 100 * np.mean(np.abs(predictions - y_true)) / np.mean(y_true)
            print(f"{percent_error}%")
            print(f"Val loss: {val_loss:.4f}")
        return predictions, y_true
    
    def predict(self, X):
        self.eval()
        with torch.no_grad():
            prediction = self.forward(X)
        return prediction
    
def get_data(path):
    #read path
    df = pd.read_csv(path)
    #create the tommorrow column - this will be y_true
    df['Next'] = df['Close'].shift(-1)
    #remove nan values
    df = df.dropna().copy()

    #seperate X and y_true
    X_df = df
    y_df = df['Next']
    del X_df['Next']

    #convert to np array
    X = X_df.to_numpy()
    y = y_df.to_numpy()

    # reshape y_train to fit model with shape (2767,1) instead of (2767,)
    y = y.reshape(-1, 1) # infer the first shape, set second shape to 1

    return X, y

def prep_data(X, y, *, save):
    if save:
        scaler_X = MinMaxScaler(feature_range=(-1, 1))
        scaler_y = MinMaxScaler(feature_range=(-1, 1))
        X = scaler_X.fit_transform(X)
        y = scaler_y.fit_transform(y)
        with open('scaler_X.pkl', 'wb') as f:
            pickle.dump(scaler_X, f)
        with open('scaler_Y.pkl', 'wb') as f:
            pickle.dump(scaler_y, f)
    else:
        with open('scaler_X.pkl', 'rb') as f:
            scaler_X = pickle.load(f)
        with open('scaler_y.pkl', 'rb') as f:
            scaler_y = pickle.load(f)
        # IMPORTANT DO NOT FIT TRANSFORM BUT RATHER TRANSFORM ALL DATA TO THE TEST DATA
        X = scaler_X.transform(X)
        y = scaler_y.transform(y)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X, y)
    return dataset

def load_prediction(path):
    with open('scaler_X.pkl', 'rb') as f:
        scaler_X = pickle.load(f)
    df = pd.read_csv(path)
    X = df.to_numpy()
    # IMPORTANT DO NOT FIT TRANSFORM BUT RATHER TRANSFORM ALL DATA TO THE TEST DATA
    X = scaler_X.transform(X)
    X = torch.tensor(X, dtype=torch.float32)

    return X
# ----------------------------------------------------------
"""
#settings
epochs = 1000
batch_size = 1024

device = torch.device('cuda')

X_train, y_train = get_data('train.csv')
X_test, y_test = get_data('test.csv')
train_data = prep_data(X_train, y_train, save=True)
test_data = prep_data(X_test, y_test, save=False)

model = Stock_Model(X_train.shape[1])

loss_function = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, eps=1e-8, fused=True)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

#training loop
for epoch in range(epochs):

    #train
    model.train()
    #set / reset loss
    train_loss = 0.0

    for X_batch, y_batch in train_loader:

        # forward
        outputs = model(X_batch)

        #loss
        loss = loss_function(outputs, y_batch)
        train_loss += loss.item() * X_batch.size(0)

        #optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    #normalize over the epoch
    train_loss /= len(train_loader.dataset)

    model.evaluation(val_loader=val_loader, loss_function=loss_function)
    print(f"Epoch: {epoch+1}, Train loss: {train_loss:.4f}")

torch.save(model.state_dict(), 'stock_regression_model_weights.pth')
"""



    