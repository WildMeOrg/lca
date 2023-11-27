import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stats import intersect_stats
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

def subsample_max_df(df, n_subsample_max, random_states=(0, 1)):
    def subsample_group(g):
        return g.sample(frac=1, random_state=random_states[0]).head(n_subsample_max)

    return df.groupby('name').apply(subsample_group).sample(frac=1, random_state=random_states[1]).reset_index(drop=True)

def split_dataframe(df, ratio=0.5):
    part = df.sample(frac = ratio)
    rest_part = df.drop(part.index)
    return part, rest_part


def split_df(df, sgkf_n_splits=10, skf_n_splits=10, verbose=False, plot_dist=True, filter_min_max=True, shuffle=True, random_state=None):
    dfa_train = df.copy()

    if sgkf_n_splits > 0:
        df_comb = df.copy()
        dfa_train, dfa_new = stratified_group_split(df_comb, 'name', 'name_viewpoint', sgkf_n_splits, random_state)
        
        if verbose:
            print_div()
            print('Calculating stats for new individual subset.')
            intersect_stats(dfa_train, dfa_new, key="name_viewpoint")

        new_names = dfa_new['name'].unique()
        train_names = dfa_train['name'].unique()

        names_intersect = np.intersect1d(new_names, train_names)
        assert len(names_intersect) == 0
    else:
        dfa_new = None

    if skf_n_splits > 0:
        dfa_train, dfa_existing = stratified_split(dfa_train, 'name', skf_n_splits, random_state)

        if verbose:
            print_div()
            print('Calculating stats for existing individual subset.')
            intersect_stats(dfa_train, dfa_existing, key="name_viewpoint")
    else:
        dfa_existing

    dfa_test_val = pd.concat([dfa_existing, dfa_new])

    test_names_unique = dfa_test_val['name'].unique()
    test_names = test_names_unique[::2]  
    val_names = test_names_unique[1::2]

    dfa_test = dfa_test_val[dfa_test_val['name'].isin(test_names)]
    dfa_val = dfa_test_val[dfa_test_val['name'].isin(val_names)]


    if filter_min_max:
        print_div()
        print('Applying filter_min=True')
        _dfa_train = filter_min_df(dfa_train, 2)
        _dfa_test = filter_min_df(subsample_max_df(dfa_test, 100), 2)
        _dfa_val = filter_min_df(subsample_max_df(dfa_val, 100), 2)
    else:
        _dfa_train = dfa_train
        _dfa_test = dfa_test
        _dfa_val = dfa_val

    print_div()
    print('Calculating stats for existing combined subsets')
    intersect_stats(_dfa_train, _dfa_test, key="name_viewpoint")
    intersect_stats(_dfa_train, _dfa_val, key="name_viewpoint", a_name="train", b_name="val")

    if plot_dist:
        plot_distribution(_dfa_train, _dfa_test, _dfa_val)

    return _dfa_train, _dfa_test, _dfa_val

def stratified_group_split(df, group_col, stratify_col, n_splits, random_state):
    df_comb = df.copy()
    dfg = df_comb.groupby(group_col)[group_col].count().sort_values(ascending=False)
    df_comb['image_count'] = df_comb[group_col].map(dfg)

    X = df_comb.index.values
    y = df_comb['image_count']
    groups = df_comb[group_col]
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    for i, (train_index, test_index) in enumerate(skf.split(X, y, groups)):
        break

    dfa_train = df_comb.iloc[train_index]
    dfa_new = df_comb.iloc[test_index]
    
    return dfa_train, dfa_new

def stratified_split(df, group_col, n_splits, random_state):
    df_comb = df.copy()
    dfg = df_comb.groupby(group_col)[group_col].count().sort_values(ascending=False)
    df_comb['image_count'] = df_comb[group_col].map(dfg)

    X = df_comb.index
    y = df_comb['image_count']
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        break

    dfa_train = df_comb.iloc[train_index]
    dfa_existing = df_comb.iloc[test_index]

    return dfa_train, dfa_existing

def filter_min_df(df, min_count):
    return df.groupby('name').filter(lambda g: len(g) >= min_count)

def plot_distribution(train_df, test_df, val_df):
    fig, ax = plt.subplots()
    train_df['species'].value_counts().plot(kind='bar', ax=ax, label="Train")
    test_df['species'].value_counts().plot(kind='bar', ax=ax, label="Test", color='orange')
    val_df['species'].value_counts().plot(kind='bar', ax=ax, label="Val",color='green')
    ax.legend()

    print_div()
    print('Train: ')
    print_group_stats(train_df)
    print()
    print('Test: ')
    print_group_stats(test_df)
    print()
    print('Val: ')
    print_group_stats(val_df)

def print_group_stats(df):
    df_annot_counts = df['species'].value_counts(ascending=True)
    df_name_counts = df.groupby('species')['name'].nunique()
    df_stat = pd.concat([df_annot_counts, df_name_counts], axis=1)
    print(df_stat)

def print_div():
    print("===================================")