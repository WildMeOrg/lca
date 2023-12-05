
import json
import pandas as pd
import glob


def load_json(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data

def write_json(data, out_path):
    json_object = json.dumps(data, indent=4)
    with open(out_path, "w") as outfile:
        outfile.write(json_object)
        
def export_annos(dfa, dfi, out_path):
    print('out_path', out_path)
    print('shapes: ', dfa.shape, dfi.shape)
    annos_list = dfa.to_dict(orient='records')
    images_list = dfi.to_dict(orient='records')
    
    print('len(images_list)', len(images_list))
    print('len(annos_list)', len(annos_list))


    data = {
        'info':{},
        'licenses':[],
        'images':images_list,
        'annotations':annos_list,
        'parts':[]
           }
    write_json(data, out_path)

def print_div():
    print()
    print('-'*50)
    print()

def join_without_intersection(df, dfs, cols, left_on, right_on):
    dfs = dfs[cols]
    return (df.drop(dfs.columns, axis=1, errors="ignore")).merge(dfs, left_on=left_on, right_on=right_on, how='left')

def final_join(df_tr, dfa, dfi, df):
    # Rename merged keys that originally changed names. These keys will be used for reference by MiewID
    dfa_uuids = df_tr['uuid_x'].unique()
    dfi_uuids = df_tr['uuid_y'].unique()

    dfa_tr = dfa[dfa['uuid'].isin(dfa_uuids)]
    dfi_tr = dfi[dfi['uuid'].isin(dfi_uuids)]

    merge_cols = ['uuid_x', 'name_viewpoint', 'species_viewpoint', 'species', 'uuid_y', 'viewpoint']
    # dfa_tr = dfa_tr.merge(df[merge_cols], left_on='uuid', right_on='uuid_x', how='left').drop('uuid_x', 1)
    dfa_tr = join_without_intersection(dfa_tr, df, merge_cols, left_on='uuid', right_on='uuid_x').drop('uuid_x', 1)
    dfa_tr['image_uuid'] = dfa_tr['uuid_y']

    # merge_cols = ['uuid_y', 'file_path']
    # dfi_tr = dfi_tr.merge(df[merge_cols], left_on='uuid', right_on=merge_cols[0], how='left').drop(merge_cols[0], 1)

    merge_cols = ['uuid_x', 'bbox']
    # dfa_tr = dfa_tr.merge(df[merge_cols], left_on='uuid', right_on='uuid_x', how='left').drop('uuid_x', 1)
    # dfa_tr['bbox'] = dfa_tr['bbox_y']
    # dfa_tr = dfa_tr.drop('bbox_y', 1)
    dfa_tr = join_without_intersection(dfa_tr, df, merge_cols, left_on='uuid', right_on='uuid_x').drop('uuid_x', 1)
    return dfa_tr, dfi_tr

def assign_viewpoint(viewpoint, excluded_viewpoints):
    if viewpoint is None:
        return None
    if viewpoint in excluded_viewpoints:
        return None
    if "left" in viewpoint:
        return "left"
    elif "right" in viewpoint:
        return "right"
    else:
        return None
    
def assign_viewpoints(df, excluded_viewpoints):
    for index, row in df.iterrows():
            df.at[index, 'viewpoint'] = assign_viewpoint(row["viewpoint"], excluded_viewpoints)
    
    # Filter out rows with NaN in the 'viewpoint' column
    df = df[~df['viewpoint'].isna()]
    return df

def filter_by_csv(df, csv_folder, names=['annotation_uuid', 'species', 'viewpoint', 'name_uuid', 'name', 'date'], merge_cols=['annotation_uuid', 'date']):
    # Filter by csv. Can support multiple csv files, all in the same folder

    
    # Load CSV files
    dfs = []
    for file_path in glob.glob(f'{csv_folder}/*'):
        _dfs = pd.read_csv(file_path, names=names)
        dfs.append(_dfs)
    # Concatenate and drop duplicates based on 'annotation_uuid'
    dfs = pd.concat(dfs)
    dfs = dfs.drop_duplicates(subset=['annotation_uuid'])


    # Keep only rows with UUIDs present in the concatenated DataFrame
    keep_uuids = set(dfs['annotation_uuid'].unique())
    df = df[df['uuid_x'].isin(keep_uuids)]
    df = df.reset_index(drop=True)

    # Merge additional information from the concatenated DataFrame
    # df = df.merge(dfs[['annotation_uuid', 'date']], left_on='uuid_x', right_on='annotation_uuid', how='left')      
    df = join_without_intersection(df, dfs, merge_cols, left_on='uuid_x', right_on='annotation_uuid')

    return df
