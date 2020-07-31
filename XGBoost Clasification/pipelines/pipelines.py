import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tqdm import tqdm


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
        return X  # [self.features_names]


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
        X = X.progress_apply(lambda column: self.merge_other(column), axis=0)
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

