import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, f_classif
from tqdm import tqdm
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score,  f1_score, plot_roc_curve
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import GridSearchCV
import re



class FeatureSelector(BaseEstimator, TransformerMixin):

    def __init__(self, features_type):
        self.features_type = features_type
        self.features_len = int
        self.features_names = []

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # dypes_dict = {'categorical': ['object'],
        #               'numerical': ['int64', 'float64', 'int']}
        # self.features_names = X.select_dtypes(include=dypes_dict.get(self.features_type)).columns

        self.features_names = X.columns
        self.features_len = len(self.features_names)
        return X#[self.features_names]


class FindOutlier(LocalOutlierFactor):

    def transform(self, X, y=None):
        # pred = self.fit_predict(X)
        lof = self.fit(X)
        print(X.shape)
        pred = lof.negative_outlier_factor_
        X = np.concatenate([X, pred[:, None]], axis=1)
        print(X.shape)
        return X


class CategoryMerger(BaseEstimator, TransformerMixin):

    def __init__(self, min_records):
        self.min_records = min_records
        self.high_freq = {}

    def high_freq_categories(self, column):
        counts = column.value_counts()
        list_high_freq = set(column.loc[counts[column].values > self.min_records]) if not None else []
        self.high_freq.update({column.name: list_high_freq})
        return column

    def merge_other(self, column):
        merged_column = column.copy()
        merged_column.loc[~merged_column.isin(self.high_freq.get(merged_column.name))] = 'other'
        return merged_column

    def fit(self, X, y=None):
        tqdm.pandas(desc='fit - high freq categories ')
        X.progress_apply(lambda column: self.high_freq_categories(column), axis=0)
        return self

    def transform(self, X, y=None):
        tqdm.pandas(desc='transform')
        X = X.progress_apply(lambda column: self.merge_other(column),  axis=0)
        return X.values

from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
def pipeline():

    numerical_steps = Pipeline(steps=[('num_selector', FeatureSelector('numerical')),
                                      ('std_scaler', StandardScaler()),
                                      ('outlier', FindOutlier(n_neighbors=4, metric='chebyshev')),
                                      # ('feature_selecttor',  SelectKBest(f_classif, k=4)),
                                      ])

    categorical_steps = Pipeline(steps=[('cat_selector', FeatureSelector('categorical')),
                                        ('merge_categories', CategoryMerger(min_records=5)),
                                        ('one_hot_encoder', OneHotEncoder(sparse=False))
                                        ])

    data_pipeline2 = FeatureUnion(transformer_list=[('numerical_pipeline', numerical_steps),
                                                   ('categorical_pipeline', categorical_steps),
                                                   ])



    data_pipeline = ColumnTransformer(transformers=[
        ('numerical_pipeline', numerical_steps, selector(dtype_exclude="object")),
        ('categorical_pipeline', categorical_steps, selector(dtype_include="object"))
    ])

    return data_pipeline


def eval_clf(classifiers, X_train, X_test, y_train, y_test):
    for i, classifier in enumerate(classifiers):
        clf_name = str(classifier).split('(')[0]
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        print('{} f1 {:0.3f} recall {:0.3f} score'.format(clf_name,
                                                          f1_score(y_test, y_pred, average='weighted'),
                                                          recall_score(y_test, y_pred, average='weighted')))
        if i == 0:
            rfc_disp = plot_roc_curve(classifier, X_test, y_test, name=clf_name)
            ax = plt.gca()
        else:
            plot_roc_curve(classifier, X_test, y_test, ax=ax, alpha=0.8, name=clf_name)#.plot(ax=ax, alpha=0.8)
    plt.rcParams["figure.figsize"] = (40, 10)
    plt.rcParams["font.size"] =22
    plt.rcParams.update({'font.size': 22})
    plt.show()

def convert_X_to_df(data_pipeline, X):
    categorical_features_names = data_pipeline.named_transformers_['categorical_pipeline'][
        'cat_selector'].features_names
    new_categorical_features_names = data_pipeline.named_transformers_['categorical_pipeline'][
        'one_hot_encoder'].get_feature_names(categorical_features_names)

    numeric_names = data_pipeline.named_transformers_['numerical_pipeline']['num_selector'].features_names.to_list()
    numeric_names.append('outliers')
    after_pipeline_columns_names = np.concatenate([numeric_names, new_categorical_features_names])
    return pd.DataFrame(X, columns=after_pipeline_columns_names)


class CustomGridSearch(GridSearchCV):
    def create_report(self):

        print(self.best_params_)
        print(self.best_score_)
        cv_results = self.cv_results_.copy()

        [cv_results.pop(x) for x in list(cv_results.keys()) if
         not re.match(f'mean_|param_classifier__|rank_test_{self.refit}', x)]
        df_cv_results = pd.DataFrame(cv_results).sort_values(f'rank_test_{self.refit}')
        df_cv_results.columns = df_cv_results.columns.str.replace('mean_|test_', '')
        return  self.best_estimator_, df_cv_results.style.bar(color='#d65f5f').format("{:.3f}")


