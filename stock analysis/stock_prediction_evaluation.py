from stock_regression import Stock_Model
import stock_regression
import torch
import pickle
from torch.utils.data import DataLoader

#settings
batch_size = 1024

#USE THE SAME SCALING AS TEST DATA
with open('scaler_X.pkl', 'rb') as f:
    scaler_X = pickle.load(f)
with open('scaler_y.pkl', 'rb') as f:
    scaler_y = pickle.load(f)


#use if validation data does not exist - simply to make a prediction
X = stock_regression.load_prediction('validation.csv')

model = Stock_Model(X.shape[1])
model.load_state_dict(torch.load("stock_regression_model_weights.pth"))

prediction_scaled = model.predict(X)
prediction = scaler_y.inverse_transform(prediction_scaled)

print(prediction)

"""
#use if validation data does exist - test if works
X_val, y_val = stock_regression.get_data('validation.csv')
val_data = stock_regression.prep_data(X_val, y_val, save=False)
val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)

predictions, y_true = model.evaluation(val_loader, torch.nn.MSELoss())

print(predictions)

print(y_true)
"""