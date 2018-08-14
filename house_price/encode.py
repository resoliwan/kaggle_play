import pandas as pd
import numpy as np
import datetime
from sklearn import preprocessing

train_df = pd.read_csv('./data/train.csv')

# train_df.select_dtypes(include=['object']).columns
# df = pd.DataFrame({
#   'cc': ['a', 'b', 'c', None],
#   'temp': [1, 2, 3, 4],
#   'cc2': ['a', 'b', 'c', None],
#   })

def create_df_cate_to_numeric(df):
  result_df = df.select_dtypes(exclude=['object']).copy()
  obj_df = df.select_dtypes(include=['object'])
  for column in obj_df.columns:
    result_df['code_' + column] = df[column].astype('category').cat.codes
  return result_df
