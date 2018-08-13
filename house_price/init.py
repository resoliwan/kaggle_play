import pandas as pd
import numpy as np
import datetime

traind_df = pd.read_csv('./data/train.csv')

test_df = pd.read_csv('./data/test.csv')
test_df.shape

y_pred = np.zeros(test_df.shape[0])
result = pd.DataFrame({'Id': test_df['Id'], 'SalePrice': y_pred})

now = datetime.datetime.today().strftime('%Y-%m-%d_%H:%M')
file_name = './submissions/result' + now + '.csv'
result.to_csv(file_name, index=False)
