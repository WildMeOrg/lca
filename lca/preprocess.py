### TODO """preprocessing scripts - filtering, subsampling, splitting"""
import json
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tools import *


def load_to_df(anno_path,format='standard'):
    data = load_json(anno_path)

    dfa = pd.DataFrame(data['annotations'])
    dfi = pd.DataFrame(data['images'])
    if format == 'standard':
        dfn = pd.DataFrame(data['individuals'])
        dfc = pd.DataFrame(data['categories'])

    dfi = dfi.drop_duplicates(subset=['uuid'])

    merge_on_uuid = 'image_uuid' in dfa.columns and 'uuid' in dfi.columns
    if merge_on_uuid:
        print('Merging on image uuid')
        df = dfa.merge(dfi, left_on='image_uuid', right_on='uuid')
    else:
        df = dfa.merge(dfi, left_on='image_id', right_on='id') 

    if format == 'standard':
        df = df.merge(dfn, left_on='individual_uuid', right_on='uuid')
        df = df.merge(dfc, left_on='category_id', right_on='id')


    print(f'** Loaded {anno_path} **')
    print('     ', f'Found {len(df)} annotations')

    return df

def filter_viewpoint_df(df, viewpoint_list):
    df = df[df['viewpoint'].isin(viewpoint_list)]
    print('     ', len(df), 'annotations remain after filtering by viewpoint list', viewpoint_list)
    return df

def filter_uuids_df(df, uuids_list):
    df = df[df['uuid_x'].isin(uuids_list)]
    print('     ', len(df), 'annotations remain after filtering by given uuids')
    return df

def filter_min_names_df(df, n_filter_min, filter_key='name_species'):
    df = df.groupby(filter_key).filter(lambda g: len(g)>=n_filter_min)
    print('     ', len(df), 'annotations remain after filtering by min', n_filter_min, 'per', filter_key)
    return df

def filter_max_df(df, n_subsample_max, filter_key='name_species'):
    df = df.groupby(filter_key, as_index=False).apply(lambda g: g.sample(frac=1, random_state=0).head(n_subsample_max)).droplevel(level=0)
    print('     ', len(df), 'annotations remain after filtering by max', n_subsample_max, 'per', filter_key)
    return df

def convert_name_to_id(names):
    le = LabelEncoder()
    names_id = le.fit_transform(names)
    return names_id

def filter_df(df, viewpoint_list, n_filter_min, n_subsample_max, embedding_uuids, filter_key='name'):
    if embedding_uuids:
        df = filter_uuids_df(df, embedding_uuids)

    if viewpoint_list:
        df = filter_viewpoint_df(df, viewpoint_list)
    
    if n_filter_min:
        df = filter_min_names_df(df, n_filter_min, filter_key=filter_key)

    if not len(df):
        raise Exception("No samples remain after filtering.")
        
    if n_subsample_max:
        df = filter_max_df(df, n_subsample_max, filter_key=filter_key)

    return df

def preprocess_data(anno_path, name_keys=['name'], convert_names_to_ids=True, viewpoint_list=None, n_filter_min=None, n_filter_max=None, images_dir=None, embedding_uuids=None, format='standard'):

    df = load_to_df(anno_path, format)

    filter_key = '__'.join(name_keys)
    df[filter_key] = df[name_keys].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
    

    df = filter_df(df, viewpoint_list, n_filter_min, n_filter_max, embedding_uuids, filter_key=filter_key)

    if convert_names_to_ids:
        names = df[filter_key].values
        names_id = convert_name_to_id(names)
        df['name_id'] = names_id
    if images_dir is not None:
        df['file_path'] = df['file_name'].apply(lambda x: os.path.join(images_dir, x))

    df = df.reset_index(drop=True)

    return df

if __name__ == "__main__":
    pass