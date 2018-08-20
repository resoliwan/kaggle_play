import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import datetime


def load_data():
  scaler = StandardScaler()
  train_df = pd.read_csv("./data/train.csv")
  Train = train_df.as_matrix()
  X_train = scaler.fit_transform(Train[:, 1:].astype(np.float32))
  y_train = Train[:, 0].astype(np.int32)
  X_valid, X_train = X_train[:5000], X_train[5000:]
  y_valid, y_train = y_train[:5000], y_train[5000:]

  test_df = pd.read_csv("./data/test.csv")
  Test = test_df.as_matrix()
  X_test = scaler.fit_transform(Test.astype(np.float32))
  return X_train, y_train, X_valid, y_valid, X_test


# y_pred = np.zeros(test_df.shape[0]).astype(np.int32)
def create_submission(y_pred):
  result = pd.DataFrame({'Label': y_pred})
  now = datetime.datetime.today().strftime('%Y-%m-%d_%H:%M')
  file_name = './submissions/result' + now + '.csv'
  result.to_csv(file_name, index=True)

