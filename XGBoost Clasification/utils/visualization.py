import warnings

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import seaborn as sns

warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)
# sns.palplot(sns.color_palette("husl"))
# sns.set_palette("husl")


def plot_counts(data, col_name):
    _, ax = plt.subplots(1, 1, figsize=(15, 5))
    p = sns.countplot(data=data, x=col_name, palette="husl")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    plt.tight_layout()
    plt.show()

    p.axes.set_title(col_name, fontsize=20)
    p.set_xlabel('')
    p.set_ylabel('Count', fontsize=15)
    _ = p.set(xticks=[])


def plot_hist(hist_1, hist_2, col):
    f, ax = plt.subplots(1, 1, figsize=(15, 5))
    a = sns.distplot(hist_1,norm_hist=False, ax=ax, bins=100, kde=True)
    b = sns.distplot(hist_2,norm_hist=False, ax=ax, bins=100, kde=True)
    ax.set_title('Total and labeled histogram for: {}'.format(col), fontsize=16)
    ax.set_xlabel('Index', fontsize=15)
    ax.set_ylabel(col, fontsize=15)
    plt.legend(['COVID Neg', 'COVID Pos'])
    plt.show()


def cat_plot(df, col):
    fig, ax1 = plt.subplots(figsize=(15, 5))

    color = 'tab:blue'
    ax1.set_title('Total count and labeled for: {}'.format(col), fontsize=16)
    ax1.set_xlabel(col, fontsize=16)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=40, ha="right")
    ax1 = sns.barplot(x=col, y='value_counts', data=df, palette='winter')
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()
    color = 'tab:red'
    sns.lineplot(x=col, y='value', hue='variable', data=df, sort=False, color=color)
    ax2.tick_params(axis='y', color=color)
    ax2.set_ylabel('prc_labeled', fontsize=16)
    ax1.set_ylabel('count', fontsize=16)


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

    fig.add_trace(go.Scatter(x=[clf.best_iteration - 1]*2,
                             y=[clf.evals_result_['validation_0'][clf_metric][clf.best_iteration],
                                clf.evals_result_['validation_1'][clf_metric][clf.best_iteration]],

                             name='Best iteration',
                             line=dict(color='purple', width=4, dash='dot')))

    fig.update_layout(title='Bias variance tradeoff curves',
                       xaxis_title='Number of iterations',
                       yaxis_title='Evaluation score')

    fig.show()
