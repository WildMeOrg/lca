
import json
import pandas as pd
import glob
import yaml


def load_json(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data

def write_json(data, out_path):
    json_object = json.dumps(data, indent=4)
    with open(out_path, "w") as outfile:
        outfile.write(json_object)
        
def export_annos(dfa, dfi, out_path):
    """
    Export annotations and images dataframes to a JSON file.

    Parameters:
    - dfa: Annotations DataFrame.
    - dfi: Images DataFrame.
    - out_path: Path to the output JSON file.

    Returns:
    - None
    """
    print('out_path:', out_path)
    print('shapes: ', dfa.shape, dfi.shape)
    
    # Convert DataFrames to dictionaries
    annos_list = dfa.to_dict(orient='records')
    images_list = dfi.to_dict(orient='records')
    
    print('len(images_list):', len(images_list))
    print('len(annos_list):', len(annos_list))

    # Create the JSON data structure
    data = {
        'info': {},
        'licenses': [],
        'images': images_list,
        'annotations': annos_list,
        'parts': []
    }

    # Write the data to the specified JSON file
    write_json(data, out_path)
    
    return

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


    merge_cols = ['uuid_x', 'name_viewpoint', 'species_viewpoint', 'species', 'uuid_y', 'viewpoint', 'name']
    
    dfa_tr = join_without_intersection(dfa_tr, df, merge_cols, left_on='uuid', right_on='uuid_x').drop('uuid_x', 1)

    dfa_tr['image_uuid'] = dfa_tr['uuid_y']


    merge_cols = ['uuid_x', 'bbox']
    dfa_tr = join_without_intersection(dfa_tr, df, merge_cols, left_on='uuid', right_on='uuid_x').drop('uuid_x', 1).drop('uuid_y', axis=1)
    return dfa_tr, dfi_tr

def assign_viewpoint(viewpoint, excluded_viewpoints):
    """
    Assign or modify viewpoint values to "right" or "left".

    Parameters:
    - viewpoint: Current viewpoint value to be assigned or modified.
    - excluded_viewpoints: List of viewpoint values to be excluded.

    Returns:
    - Assigned or modified viewpoint value.
    """

    if viewpoint is None:
        return None
    if viewpoint in excluded_viewpoints:
        return None
    if "left" in viewpoint:
        return "left"
    elif "right" in viewpoint:
        return "right"
    else:
        return viewpoint

def assign_viewpoints(df, excluded_viewpoints):
    """
    Assign or modify viewpoint values in a DataFrame based on specified rules.

    Parameters:
    - df: DataFrame containing 'viewpoint' column to be modified.
    - excluded_viewpoints: List of viewpoint values to be excluded.

    Returns:
    - DataFrame with assigned or modified 'viewpoint' values, excluding rows with NaN in 'viewpoint'.
    """
    for index, row in df.iterrows():
        df.at[index, 'viewpoint'] = assign_viewpoint(row["viewpoint"], excluded_viewpoints)
    
    # Filter out rows with NaN in the 'viewpoint' column
    df = df[~df['viewpoint'].isna()]
    return df

def filter_by_csv(df, csv_folder, 
                csv_column_names=['annotation_uuid', 'species', 'viewpoint', 'name_uuid', 'name', 'date'], 
                merge_columns=['annotation_uuid', 'date']):
    """
    Filter and merge a DataFrame using CSV files in a specified folder.

    Parameters:
    - df: DataFrame to be filtered and merged.
    - csv_folder: Folder containing CSV files for filtering and merging.
    - csv_column_names: List of column names for the CSV files.
    - merge_columns: List of columns for merging DataFrames.

    Returns:
    - Merged DataFrame.
    """

    # Load CSV files
    csv_dataframes = []
    for file_path in glob.glob(f'{csv_folder}/*'):
        csv_dataframe = pd.read_csv(file_path, names=csv_column_names)
        csv_dataframes.append(csv_dataframe)

    # Concatenate and drop duplicates based on 'annotation_uuid'
    concatenated_csv = pd.concat(csv_dataframes)
    print("Total annotations in all CSV files:", len(concatenated_csv))
    concatenated_csv = concatenated_csv.drop_duplicates(subset=['annotation_uuid'])
    print("Unique annotations in all CSV files:", len(concatenated_csv))

    # Keep only rows with UUIDs present in the concatenated DataFrame
    keep_uuids = set(concatenated_csv['annotation_uuid'].unique())
    df = df[df['uuid_x'].isin(keep_uuids)]
    df = df.reset_index(drop=True)
    print("Annotations after CSV merge:", len(df))

    # Merge DataFrames using 'join_without_intersection' function (assuming this function is defined elsewhere)
    df = join_without_intersection(df, concatenated_csv, merge_columns, left_on='uuid_x', right_on='annotation_uuid')

    return df



def read_yaml_config(file_path, species):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)

    species_config = config.get(species)

    if species_config:
        # Replace placeholders in the config
        species_config = {k: v.format(**species_config) if isinstance(v, str) else v for k, v in species_config.items()}
        return species_config
    else:
        raise ValueError(f"Configuration for species '{species}' not found.")
    

def add_image_count(df, group_col):
    dfg = df.groupby(group_col)[group_col].count().sort_values(ascending=False)
    df['image_count'] = df[group_col].map(dfg)

    return df


def subsample_max_df(df, group_col='name', n_subsample_max=4, random_states=(0, 1)):
    """
    Subsample a DataFrame by groups, ensuring a maximum number of samples per group.

    Parameters:
    - df: DataFrame to be subsampled.
    - group_col: Column used for grouping (default: 'name').
    - n_subsample_max: Maximum number of samples per group (default: 4).
    - random_states: Tuple of random states for shuffling and sampling (default: (0, 1)).

    Returns:
    - Subsampled DataFrame.
    """
    def subsample_group(g):
        return g.sample(frac=1, random_state=random_states[0]).head(n_subsample_max)

    return df.groupby(group_col).apply(subsample_group).sample(frac=1, random_state=random_states[1]).reset_index(drop=True)

def filter_min_df(df, key='name', min_count=2):
    """
    Filter a DataFrame to keep only rows with a minimum count of a specified key.

    Parameters:
    - df: DataFrame to be filtered.
    - key: Column name used for counting (default: 'name').
    - min_count: Minimum count required for a key to be retained (default: 2).

    Returns:
    - Filtered DataFrame.
    """
    return df.groupby(key).filter(lambda g: len(g) >= min_count)

def apply_filters(dataframe, key, max_df, min_df):
    """
    Apply maximum and minimum filtering to a DataFrame based on key counts.

    Parameters:
    - dataframe: DataFrame to be filtered.
    - key: Column name used for counting.
    - max_df: Maximum count to apply subsampling (None by default, indicating no maximum).
    - min_df: Minimum count to apply filtering (None by default, indicating no minimum).

    Returns:
    - Filtered DataFrame.
    """
    if max_df is not None:
        dataframe = subsample_max_df(dataframe, key, max_df)
    if min_df is not None:
        dataframe = filter_min_df(dataframe, key, min_df)
    return dataframe
