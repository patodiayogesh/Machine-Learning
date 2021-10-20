
class NB:
    """
    Naive Bayes Estimator

    Attributes
    __________
    f1: numpy array containing probability vectors of label 1
    f0: numpy array containing probability vectors of label 0
    """

    def __init__(self):
        self.f1 = {}
        self.f0 = {}

    def get_feature(self,df):
        """
        Function to calculate count probabilities
        of each feature for each label

        :param df: {pandas DataFrame} Input Training Data
        :return: {dict} Count probability of features
        """

        f = {}
        l = len(df)
        for col in df.columns:
            a = df[col].value_counts()

            f[col] = a.div(l)
        return f

    def fit(self,train_data, y_col_name):
        """
        Build Naive Bayes Classifier

        :param train_data: {pandas DataFrame} Input Training Data
        :param y_col_name: {str} Label Column name
        :return: Naive Bayes fitted classifier
        """

        df_1 = train_data.loc[train_data[y_col_name] == 1]
        df_0 = train_data.loc[train_data[y_col_name] == 0]

        self.f1 = self.get_feature(df_1)
        self.f0 = self.get_feature(df_0)

        return self

    def calculate_prob(self, row):
        """
        Function to calculate probability of each feature
        for test data point

        :param row: {pandas Series} Test feature vector
        :return: {int} Predicted Label
        """

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
        """
        Function to predict probabilities of input dataset

        :param df_test: {pandas DataFrame} Test Dataset
        :return: {pandas Series} Predicted Labels of each datapoint
        """

        df_test['p'] = df_test.apply(lambda x: self.calculate_prob(x), axis=1)
        return df_test.loc[:, df_test.columns == 'p']


