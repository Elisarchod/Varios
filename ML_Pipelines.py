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

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        dypes_dict = {'categorical': ['object'],
                      'numerical': ['int64', 'float64']}

        _feature_names = X.select_dtypes(include=dypes_dict.get(self.features_type)).columns
        return X[_feature_names]


class FindOutlier(LocalOutlierFactor):

    def transform(self, X, y=None):
        pred = self.fit_predict(X)
        return np.concatenate([X, pred[:, None]], axis=1)


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
        X.apply(lambda column: self.high_freq_categories(column), axis=0)
        return self

    def transform(self, X, y=None):

        print(f'shape pre transform {X.shape}')
        # tqdm.pandas(desc='transform_apply')
        X = X.apply(lambda column: self.merge_other(column),  axis=0)
        print(f'shape after transform {X.shape}')
        return X.values


def pipeline():
    categorical_pipeline = Pipeline(steps=[('cat_selector', FeatureSelector('categorical')),
                                           ('merge_categories', CategoryMerger(min_records=5)),
                                           ('one_hot_encoder', OneHotEncoder(sparse=False))])

    numerical_pipeline = Pipeline(steps=[('num_selector', FeatureSelector('numerical')),
                                         ('std_scaler', StandardScaler()),
                                         ('outlier', FindOutlier(n_neighbors=4, metric='chebyshev')),
                                         # ('feature_selecttor',  SelectKBest(f_classif, k=4)),
                                         ])

    data_pipeline = FeatureUnion(transformer_list=[('categorical_pipeline', categorical_pipeline),
                                                   ('numerical_pipeline', numerical_pipeline)])

    return data_pipeline



def bias_var_plot(clf):
    clf_metric = clf.kwargs['eval_metric']
    x_axis_len = np.arange(1, len(clf.evals_result_['validation_0'][clf_metric])-1)
    # Create traces
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_axis_len, y=clf.evals_result_['validation_0'][clf_metric],
                        mode='lines',
                        name='train set'))
    fig.add_trace(go.Scatter(x=x_axis_len, y=clf.evals_result_['validation_1'][clf_metric],
                        mode='lines',
                        name='test set'))

    fig.add_trace(go.Scatter(x=[clf.best_iteration]*2,
                             y=[clf.evals_result_['validation_0'][clf_metric][clf.best_iteration], clf.evals_result_['validation_1'][clf_metric][clf.best_iteration]],

                             name='Best iteration',
                             line=dict(color='purple', width=4, dash='dot')))

    fig.update_layout(title='Bias variance tradeoff curves',
                       xaxis_title='Number of iterations',
                       yaxis_title='Evaluation score')

    fig.show()


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



def grid_search(clf, param_grid, X_train, X_test, y_train, y_test, **kwargs):
    gscv = GridSearchCV(clf, param_grid, n_jobs=-1, cv=2, scoring=kwargs['scoring'],
                        refit=kwargs['refit'], verbose=True).fit(X_train, y_train,
                                                      eval_set=[(X_train, y_train), (X_test, y_test)],
                                                      early_stopping_rounds=kwargs['EARLY_STOP'],
                                                      verbose=kwargs['VERBOSE'])
    print(gscv.best_params_)
    print(gscv.best_score_)

    cv_results = gscv.cv_results_.copy()

    [cv_results.pop(x) for x in list(cv_results.keys()) if
     not re.match(f'mean_|param_classifier__|rank_test_{gscv.refit}', x)]
    df_cv_results = pd.DataFrame(cv_results).sort_values(f'rank_test_{gscv.refit}')
    df_cv_results.columns = df_cv_results.columns.str.replace('mean_|test_', '')
    return  gscv, gscv.best_estimator_, df_cv_results.style.bar(color='#d65f5f').format("{:.3f}")


def grid_search2(clf, param_grid, X_train, X_test, y_train, y_test, **kwargs):
    gscv = GridSearchCV(clf, param_grid, n_jobs=-1, **kwargs).fit(X_train, y_train,
                                                      eval_set=[(X_train, y_train), (X_test, y_test)], **kwargs)
    print(gscv.best_params_)
    print(gscv.best_score_)

    cv_results = gscv.cv_results_.copy()

    [cv_results.pop(x) for x in list(cv_results.keys()) if
     not re.match(f'mean_|param_classifier__|rank_test_{gscv.refit}', x)]
    df_cv_results = pd.DataFrame(cv_results).sort_values(f'rank_test_{gscv.refit}')
    df_cv_results.columns = df_cv_results.columns.str.replace('mean_|test_', '')
    return gscv, gscv.best_estimator_, df_cv_results.style.bar(color='#d65f5f').format("{:.3f}")