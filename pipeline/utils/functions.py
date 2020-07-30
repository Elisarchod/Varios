from sklearn.preprocessing import LabelEncoder
import numpy as np

def merge_categories(column, min_records, encode=True):
    counts = column.value_counts()
    merged_column = column.copy()
    merged_column.loc[counts[column].values < min_records] = 'other'
    print('Number of classes after merge: {}'.format(len(merged_column.value_counts())))

    if encode:
        encoder = LabelEncoder()
        merged_column = encoder.fit_transform(merged_column)

    return merged_column

def gradient(predt, y):
    '''Compute the gradient squared log error.'''
    return (np.log1p(predt) - np.log1p(y)) / (predt + 1)

def hessian(predt, y):
    '''Compute the hessian for squzared log error.'''
    return ((-np.log1p(predt) + np.log1p(y) + 1) / np.power(predt + 1, 2))

def squared_log(predt, y):
    '''Squared Log Error objective. A simplified version for RMSLE used as
    objective function.
    '''
    predt[predt < -1] = -1 + 1e-6
    grad = gradient(predt, y)
    hess = hessian(predt, y)
    return 0, 0