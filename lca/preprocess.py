### TODO """preprocessing scripts - filtering, subsampling, splitting"""
import json
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tools import *


logger = logging.getLogger('lca')

def load_to_df(anno_path,format='standard', print_func=print):
    data = load_json(anno_path)

    dfa = pd.DataFrame(data['annotations'])
    dfi = pd.DataFrame(data['images'])
    if format == 'standard':
        dfn = pd.DataFrame(data['individuals'])
        dfc = pd.DataFrame(data['categories'])

    dfi = dfi.drop_duplicates(subset=['uuid'])

    merge_on_uuid = 'image_uuid' in dfa.columns and 'uuid' in dfi.columns
    if merge_on_uuid:
        print_func('Merging on image uuid')
        df = dfa.merge(dfi, left_on='image_uuid', right_on='uuid', suffixes=('', '_y'))
    else:
        df = dfa.merge(dfi, left_on='image_id', right_on='id', suffixes=('', '_y')) 

    if format == 'standard':
        df = df.merge(dfn, left_on='individual_uuid', right_on='uuid', suffixes=('', '_y'))
        df = df.merge(dfc, left_on='category_id', right_on='id', suffixes=('',   '_y'))


    print_func(f'** Loaded {anno_path} **')
    print_func(f'      Found {len(df)} annotations')

    return df

def filter_field_df(df, field, field_values, print_func=print):
    """Generic field filtering function with substring matching."""
    def contains_any_filter(value):
        """Check if any filter_value is contained in the value as substring."""
        if pd.isna(value):
            return False
        value_str = str(value)
        return any(str(filter_val) in value_str for filter_val in field_values)
    
    if field in df.columns:
        df = df[df[field].apply(contains_any_filter)]
        print_func(f'      {len(df)} annotations remain after filtering by {field} list {field_values} (substring match)')
    else:
        name_field = f"name_{field}"
        if name_field in df.columns:
            df = df[df[name_field].apply(contains_any_filter)]
            print_func(f'      {len(df)} annotations remain after filtering by {name_field} list {field_values} (substring match)')
        else:
            print_func(f'      WARNING: Field {field} not found for filtering')
    return df

def filter_uuids_df(df, uuids_list, id_key='uuid', print_func=print):
    # print_func(uuids_list)
    print_func(df.columns)
    df = df[df[id_key].isin(uuids_list)]
    print_func(f'      {len(df)} annotations remain after filtering by given uuids')
    return df

def filter_min_names_df(df, n_filter_min, filter_key='name_species', print_func=print):
    df = df.groupby(filter_key).filter(lambda g: len(g)>=n_filter_min)
    print_func(f'      {len(df)} annotations remain after filtering by min {n_filter_min} per {filter_key}')
    return df

def filter_max_df(df, n_subsample_max, filter_key='name_species', print_func=print):
    df = df.groupby(filter_key, as_index=False).apply(lambda g: g.sample(frac=1, random_state=0).head(n_subsample_max)).droplevel(level=0)
    print_func(f'      {len(df)} annotations remain after filtering by max {n_subsample_max} per {filter_key}')
    return df

def convert_name_to_id(names):
    le = LabelEncoder()
    names_id = le.fit_transform(names)
    return names_id

def filter_df(df, n_filter_min, n_subsample_max, embedding_uuids, filter_key='name', id_key='uuid', field_filters=None, print_func=print):
    if embedding_uuids:
        df = filter_uuids_df(df, embedding_uuids, id_key, print_func=print_func)

    # Generic field filtering (includes viewpoint)
    if field_filters:
        for field, values in field_filters.items():
            if values:  # Only filter if values are provided
                df = filter_field_df(df, field, values, print_func=print_func)
    
    if n_filter_min:
        df = filter_min_names_df(df, n_filter_min, filter_key=filter_key, print_func=print_func)

    if not len(df):
        raise EmptyDataframeException("No samples remain after filtering.")
        
    if n_subsample_max:
        df = filter_max_df(df, n_subsample_max, filter_key=filter_key, print_func=print_func)

    return df

def preprocess_data(anno_path, name_keys=['name'], 
                    convert_names_to_ids=True, 
                    n_filter_min=None, 
                    n_filter_max=None, 
                    images_dir=None, 
                    embedding_uuids=None, 
                    id_key='uuid', 
                    format='standard',
                    field_filters=None,
                    print_func=print):

    df = load_to_df(anno_path, format, print_func=print_func)

    filter_key = '__'.join(name_keys)
    df[filter_key] = df[name_keys].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
    

    df = filter_df(df, n_filter_min, n_filter_max, embedding_uuids, filter_key=filter_key, id_key=id_key, field_filters=field_filters, print_func=print_func)

    if convert_names_to_ids:
        names = df[filter_key].values
        names_id = convert_name_to_id(names)
        df['name_id'] = names_id
    if images_dir is not None:
        df['file_path'] = df['file_name'].apply(lambda x: os.path.join(images_dir, x))

    df = df.reset_index(drop=True)

    return df


def save_data(df, output_path, id_key='uuid'):
    df_annotations_fields = [id_key, 'image_uuid', 'bbox', 'theta', 'name', 'viewpoint', 'name_viewpoint','category_id']
    df_annotations = df[df_annotations_fields]
    # df_annotations = df_annotations.rename(columns={id_key: 'uuid'})


    df_images_fields = ['image_uuid', 'file_name', 'height', 'width', 'date_captured']
    df_images_fields= df.columns.intersection(df_images_fields)
    df_images = df[df_images_fields].drop_duplicates(keep='first').reset_index(drop=True)
    df_images = df_images.rename(columns={'image_uuid': 'uuid'})


    df_categories_fields = ['category_id', 'species']
    df_categories = df[df_categories_fields].drop_duplicates(keep='first').reset_index(drop=True)
    df_categories = df_categories.rename(columns={'category_id': 'id'})

    result_dict = {'categories': df_categories.to_dict(orient='records'),
                   'images': df_images.to_dict(orient='records'),
                   'annotations': df_annotations.to_dict(orient='records')}

    with open(output_path, 'w') as f:
        json.dump(result_dict, f, indent=4)
        print('Data is saved to:', output_path)
    return df

if __name__ == "__main__":
    pass