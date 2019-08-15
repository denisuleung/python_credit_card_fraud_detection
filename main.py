import numpy as np
import pandas as pd
import os
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


class ExcelIO:
    def __init__(self, raw_df=None, path=None):
        self.raw_df = raw_df
        self.path = path

    def import_csv(self):
        self.path = os.path.dirname(os.path.abspath(__file__)) + "/"
        self.raw_df = pd.read_csv(self.path + "creditcard.csv")


excel = ExcelIO()
excel.import_csv()


# 1. Descriptive Statistics and Chart
class BasicInfoProvider:
    def __init__(self, raw_df, pivot=None, bar_chart=None, melted=None):
        self.raw_df = raw_df
        self.pivot = pivot
        self.bar_chart = bar_chart
        self.melted = melted

    def show_avg_table(self):
        # header = list(excel.raw_df).remove('Amount'), 'Time', 'Class'])
        header = [elem for elem in list(excel.raw_df) if elem not in ['Amount', 'Time', 'Class']]

        self.pivot = pd.pivot_table(self.raw_df, values=header, columns=['Class'], aggfunc=[np.mean, np.std])
        self.pivot.reset_index(level=0, inplace=True)
        self.pivot.columns = ['index', 'mean_0', 'mean_1', 'std_0', 'std_1']
        # self.melted = pd.melt(self.pivot, id_vars=['mean_0'], value_vars=['mean_1'])

    def show_difference_by_chart(self):
        self.bar_chart = sns.barplot(x='index', y='mean_0', data=self.pivot, label="mean_0", color="b")
        # sns.set_color_codes("pastel")
        # sns.barplot(x='index', y='mean_0', data=self.pivot, label="mean_0", color="b")
        # sns.set_color_codes("muted")
        # sns.barplot(x='index', y='mean_1', data=self.pivot, label="mean_1", color="b")

    def main(self):
        self.show_avg_table()
        # self.show_difference_by_chart()  # the chart can't show the difference


info = BasicInfoProvider(excel.raw_df)
info.main()


# 2. Split to Train and Test Data Set
class Split:
    def __init__(self, df=None, y=None, x=None, train_x=None, val_x=None, train_y=None, val_y=None):
        self.df = df
        self.y, self.x = y, x
        self.train_x, self.val_x = train_x, val_x
        self.train_y, self.val_y = train_y, val_y

    def set_parameter(self):
        self.y = self.df.Class
        self.x = self.df.loc[:, (self.df.columns != 'Class')]

    def split_train_valid(self):
        self.train_x, self.val_x, self.train_y, self.val_y = train_test_split(self.x, self.y, train_size=0.9, test_size=0.1, random_state=0)

    def main(self):
        self.set_parameter()
        self.split_train_valid()


split_df = Split(info.raw_df)
split_df.main()


# 3. Modeling
class Model:
    def __init__(self, train_x=None, train_y=None, model=None):
        self.train_x = train_x
        self.train_y = train_y
        self.model = model

    def regress(self):
        self.model = RandomForestRegressor(n_estimators=20)
        self.model.fit(self.train_x, self.train_y)


model_df = Model(split_df.train_x, split_df.train_y)
model_df.regress()


class Evaluation:
    def __init__(self, model=None, eva_df=None, compare_raw=None):
        self.model = model
        self.eva_df = eva_df
        self.compare_raw = compare_raw

    def predict_result(self):
        self.eva_df['Predicted_Class'] = self.model.predict(self.eva_df)
        self.eva_df['Predicted_Class'] = self.eva_df['Predicted_Class'].round(0).astype(int)

    def compare_result(self):
        self.compared_result = pd.merge(self.eva_df, self.compare_raw['Class'], left_index=True, right_index=True)

    def value_count_for_column(self):
        self.compared_result['Predicted_Class'].value_counts()
        self.compared_result['Class'].value_counts()

    def main(self):
        self.predict_result()
        self.compare_result()
        self.value_count_for_column()


evaluated_df = Evaluation(model_df.model, split_df.val_x, info.raw_df)
evaluated_df.main()

fraud_df = evaluated_df.compared_result[(evaluated_df.compared_result['Predicted_Class'] == 1) | (evaluated_df.compared_result['Class'] == 1)]

# 4. KPI of Effectiveness
# 5. (Additional) By K-Folds
