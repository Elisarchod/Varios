import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, f1_score, plot_roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder


def merge_categories(column, min_records, encode=True):
    counts = column.value_counts()
    merged_column = column.copy()
    merged_column.loc[counts[column].values < min_records] = 'other'
    print('Number of classes after merge: {}'.format(len(merged_column.value_counts())))

    if encode:
        encoder = LabelEncoder()
        merged_column = encoder.fit_transform(merged_column)

    return merged_column


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
            plot_roc_curve(classifier, X_test, y_test, ax=ax, alpha=0.8, name=clf_name)  # .plot(ax=ax, alpha=0.8)
    plt.rcParams["figure.figsize"] = (40, 10)
    plt.rcParams["font.size"] = 22
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
        print(f'Best params for grid search {self.best_params_} with metric {self.refit} score {self.best_score_}')
        cv_results = self.cv_results_.copy()

        [cv_results.pop(x) for x in list(cv_results.keys()) if
         not re.match(f'mean_|param_classifier__|rank_test_{self.refit}', x)]
        df_cv_results = pd.DataFrame(cv_results).sort_values(f'rank_test_{self.refit}')
        df_cv_results.columns = df_cv_results.columns.str.replace('mean_|test_', '')
        return self.best_estimator_, df_cv_results.style.bar(color='#d65f5f').format("{:.3f}")


def gradient(predt, y):
    """Compute the gradient squared log error."""
    return (np.log1p(predt) - np.log1p(y)) / (predt + 1)


def hessian(predt, y):
    """Compute the hessian for squared log error."""
    return (-np.log1p(predt) + np.log1p(y) + 1) / np.power(predt + 1, 2)


def squared_log(predt, y):
    """Squared Log Error objective. A simplified version for RMSLE used as
    objective function.
    """
    predt[predt < -1] = -1 + 1e-6
    grad = gradient(predt, y)
    hess = hessian(predt, y)
    return grad, hess
