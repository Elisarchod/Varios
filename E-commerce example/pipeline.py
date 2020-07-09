import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline


# Custom Transformer that extracts columns passed as argument to its constructor
class FeatureSelector(BaseEstimator, TransformerMixin):
    # Class Constructor
    def __init__(self, feature_names):
        self._feature_names = feature_names

        # Return self nothing else to do here

    def fit(self, X, y=None):
        return self

        # Method that describes what we need this transformer to do

    def transform(self, X, y=None):
        return X[self._feature_names]


# Custom transformer that breaks dates column into year, month and day into separate columns and
# converts certain features to binary
class CategoricalTransformer(BaseEstimator, TransformerMixin):
    # Class constructor method that takes in a list of values as its argument
    def __init__(self, use_dates=['year', 'month', 'day']):
        self._use_dates = use_dates

    # Return self nothing else to do here
    def fit(self, X, y=None):
        return self

    # Helper function to extract year from column 'dates'
    def get_year(self, obj):
        return str(obj)[:4]

    # Helper function to extract month from column 'dates'
    def get_month(self, obj):
        return str(obj)[4:6]

    # Helper function to extract day from column 'dates'
    def get_day(self, obj):
        return str(obj)[6:8]

    # Helper function that converts values to Binary depending on input
    def create_binary(self, obj):
        if obj == 0:
            return 'No'
        else:
            return 'Yes'

    # Transformer method we wrote for this transformer
    def transform(self, X, y=None):
        # Depending on constructor argument break dates column into specified units
        # using the helper functions written above
        for spec in self._use_dates:
            exec("X.loc[:,'{}'] = X['date'].apply(self.get_{})".format(spec, spec))
        # Drop unusable column
        X = X.drop('date', axis=1)

        # Convert these columns to binary for one-hot-encoding later
        X.loc[:, 'waterfront'] = X['waterfront'].apply(self.create_binary)

        X.loc[:, 'view'] = X['view'].apply(self.create_binary)

        X.loc[:, 'yr_renovated'] = X['yr_renovated'].apply(self.create_binary)
        # returns numpy array
        return X.values


# Custom transformer we wrote to engineer features ( bathrooms per bedroom and/or how old the house is in 2019  )
# passed as boolen arguements to its constructor
class NumericalTransformer(BaseEstimator, TransformerMixin):
    # Class Constructor
    def __init__(self, bath_per_bed=True, years_old=True):
        self._bath_per_bed = bath_per_bed
        self._years_old = years_old

    # Return self, nothing else to do here
    def fit(self, X, y=None):
        return self

        # Custom transform method we wrote that creates aformentioned features and drops redundant ones

    def transform(self, X, y=None):
        # Check if needed
        if self._bath_per_bed:
            # create new column
            X.loc[:, 'bath_per_bed'] = X['bathrooms'] / X['bedrooms']
            # drop redundant column
            X.drop('bathrooms', axis=1)
        # Check if needed
        if self._years_old:
            # create new column
            X.loc[:, 'years_old'] = 2019 - X['yr_built']
            # drop redundant column
            X.drop('yr_built', axis=1)

        # Converting any infinity values in the dataset to Nan
        X = X.replace([np.inf, -np.inf], np.nan)
        # returns a numpy array
        return X.values

#
# # Categrical features to pass down the categorical pipeline
# cateforical_features = ['date', 'waterfront', 'view', 'yr_renovated']
#
# # Numerical features to pass down the numerical pipeline
# numerical_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
#                       'condition', 'grade', 'sqft_basement', 'yr_built']
#
# # Defining the steps in the categorical pipeline
# categorical_pipeline = Pipeline(steps=[('cat_selector', FeatureSelector(categorical_features)),
#
#                                        ('cat_transformer', CategoricalTransformer()),
#
#                                        ('one_hot_encoder', OneHotEncoder(sparse=False))])
#
# # Defining the steps in the numerical pipeline
# numerical_pipeline = Pipeline(steps=[('num_selector', FeatureSelector(numerical_features)),
#
#                                      ('num_transformer', NumericalTransformer()),
#
#                                      ('imputer', SimpleImputer(strategy='median')),
#
#                                      ('std_scaler', StandardScaler())])
#
# # Combining numerical and categorical piepline into one full big pipeline horizontally
# # using FeatureUnion
# full_pipeline = FeatureUnion(transformer_list=[('categorical_pipeline', categorical_pipeline),
#
#                                                ('numerical_pipeline', numerical_pipeline)])
#
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
#
# # Leave it as a dataframe becuase our pipeline is called on a
# # pandas dataframe to extract the appropriate columns, remember?
# X = data.drop('price', axis=1)
# # You can covert the target variable to numpy
# y = data['price'].values
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # The full pipeline as a step in another pipeline with an estimator as the final step
# full_pipeline_m = Pipeline(steps=[('full_pipeline', full_pipeline),
#
#                                   ('model', LinearRegression())])
#
# # Can call fit on it just like any other pipeline
# full_pipeline_m.fit(X_train, y_train)

# Can predict with it like any other pipeline
# y_pred = full_pipeline_m.predict(X_test)

