from encode import create_df_cate_to_numeric
from init import split_train_test
import pandas as pd
import numpy as np
import datetime
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor

train_df = pd.read_csv('./data/train.csv')
n_train_df = create_df_cate_to_numeric(train_df)
n_train_df.fillna(0, inplace=True)
X_train, X_test, y_train, y_test = split_train_test(n_train_df, test_size=0.01)


y_label = 'SalePrice'
df_y = n_train_df[y_label]
df_X = n_train_df.drop(y_label, axis=1)

grbt = GradientBoostingRegressor(
    n_estimators=1500,
    learning_rate=0.5
    )
grbt.fit(X_train, y_train)


test_df = pd.read_csv('./data/test.csv')
n_test_df = create_df_cate_to_numeric(test_df)
n_test_df.fillna(0, inplace=True)
y_pred = grbt.predict(n_test_df)
result = pd.DataFrame({'Id': test_df['Id'], 'SalePrice': y_pred})
now = datetime.datetime.today().strftime('%Y-%m-%d_%H:%M')
file_name = './submissions/result' + now + '.csv'
result.to_csv(file_name, index=False)




