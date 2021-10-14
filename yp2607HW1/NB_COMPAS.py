import pandas as pd
from sklearn.metrics import classification_report

class NB:

    def __init__(self):
        self.f1 = {}
        self.f0 = {}

    def get_feature(self,df):
        f = {}
        l = len(df)
        for col in df.columns:
            a = df[col].value_counts()

            f[col] = a.div(l)
        return f

    def fit(self,train_data, y_col_name):
        df_1 = train_data.loc[train_data[y_col_name] == 1]
        df_0 = train_data.loc[train_data[y_col_name] == 0]

        self.f1 = self.get_feature(df_1)
        self.f0 = self.get_feature(df_0)

    def calculate_prob(self, row):

        p1, p2 = 1, 1
        for index in row.index:
            try:
                p1 = p1 * self.f1[index][row[index]]
            except Exception as e:
                p1 = 0
            try:
                p2 = p2 * self.f0[index][row[index]]
            except Exception as e:
                p2 = 0
        if p1 > p2:
            return 1
        else:
            return 0

    def predict(self,df_test):
        df_test['p'] = df_test.apply(lambda x: self.calculate_prob(x), axis=1)
        return df_test.loc[:, df_test.columns == 'p']

if __name__ == '__main__':
    df = pd.read_csv('compas_dataset/propublicaTrain.csv')
    df_test = pd.read_csv('compas_dataset/propublicaTest.csv')

    y_test = df_test.loc[:, df_test.columns == 'two_year_recid']
    df_test = df_test.loc[:, df_test.columns != 'two_year_recid']

    nb_clf = NB()
    nb_clf.fit(df,'two_year_recid')
    predicted = nb_clf.predict(df_test)
    print(classification_report(y_test, predicted))


