import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

def missing_values(df):
    percent_missing = df.isnull().sum()*100/len(df)
    x = list(percent_missing)
    df_missing_values = pd.DataFrame({'column name': df.columns, 'percent missing %': x})
    return df_missing_values

def modeling(x, y, model):
    x = x
    y = y
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=99)
    # with weather condition
    if model == 'logistic':
        lr = LogisticRegression()
        model_fit = lr.fit(x_train, y_train)
        pred1 = lr.predict(x_test)
        print(model_fit)
        print("\n Accuracy of Logistic Regression is: \n", classification_report(y_test, pred1))

    else:
        rf = RandomForestClassifier()
        model_fit = rf.fit(x_train, y_train)
        pred1 = rf.predict(x_test)
        print(model_fit)
        print("\n Accuracy of Random Forest is: \n", classification_report(y_test, pred1))
