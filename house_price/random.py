from encode import create_df_cate_to_numeric
from init import split_train_test
import pandas as pd
import numpy as np
import datetime
from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import VotingClassifier
# from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_error

train_df = pd.read_csv('./data/train.csv')
n_train_df = create_df_cate_to_numeric(train_df)
n_train_df.fillna(0, inplace=True)

X_train, X_test, y_train, y_test = split_train_test(n_train_df)

# line_clf = LinearRegression()

rand_clf = RandomForestClassifier(n_estimators=1000)

# svm_clf = SVC(probability=True)

# voting_clf = VotingClassifier(
#     estimators=[('lr', line_clf), ('rnd', rand_clf), ('svm', svm_clf)],
#     voting='hard'
#     )

rand_clf.fit(X_train, y_train)


y_pred = rand_clf.predict(X_test)
mean_absolute_error(y_test, y_pred)

test_df = pd.read_csv('./data/test.csv')
n_test_df = create_df_cate_to_numeric(test_df)
n_test_df.fillna(0, inplace=True)

y_pred = rand_clf.predict(n_test_df)

result = pd.DataFrame({'Id': test_df['Id'], 'SalePrice': y_pred})

now = datetime.datetime.today().strftime('%Y-%m-%d_%H:%M')
file_name = './submissions/result' + now + '.csv'
result.to_csv(file_name, index=False)




