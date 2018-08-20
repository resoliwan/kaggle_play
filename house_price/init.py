import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split

train_df = pd.read_csv('./data/train.csv')


def split_train_test(df, test_size=0.25):
  y_label = 'SalePrice'
  df_y = df[y_label]
  df_X = df.drop(y_label, axis=1)
  return train_test_split(df_X, df_y, test_size=test_size)


test_df = pd.read_csv('./data/test.csv')
test_df.shape

y_pred = np.zeros(test_df.shape[0])
result = pd.DataFrame({'Id': test_df['Id'], 'SalePrice': y_pred})

now = datetime.datetime.today().strftime('%Y-%m-%d_%H:%M')
file_name = './submissions/result' + now + '.csv'
result.to_csv(file_name, index=False)
# kaggle competitions submit house-prices-advanced-regression-techniques -f submissions/result2018-08-13_14:24.csv -m '1'
