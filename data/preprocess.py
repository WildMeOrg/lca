from tools import load_json, filter_by_csv
import pandas as pd
from argparse import ArgumentParser
import os
import uuid
import json

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def generate_uuid_map(df, column_name):
    unique_values = df[column_name].unique()
    uuid_map = {value: str(uuid.uuid4()) for value in unique_values}
    return uuid_map

def merge_dataframes(dfa, dfi):
    if 'image_uuid' in dfa.columns and 'uuid' in dfi.columns:
        print('Merging on image uuid')
        df = dfa.merge(dfi, left_on='image_uuid', right_on='uuid')
    else:
        df = dfa.merge(dfi, left_on='image_id', right_on='id')
        images_uuids = generate_uuid_map(df, 'image_id')
        df['image_uuid'] = df['image_id'].map(images_uuids)
    return df

def add_individual_uuid(df):
    individual_ids = generate_uuid_map(df, 'name')
    df['individual_uuid'] = df['name'].map(individual_ids)
    return df

def apply_csv_filter(df, csv_dir, csv_column_names=['annotation_uuid', 'viewpoint_y'], merge_cols=['annotation_uuid', 'viewpoint_y']):

    if os.path.exists(csv_dir):
        print("Filtering_by_csv...")
        df = filter_by_csv(df, csv_dir, csv_column_names=csv_column_names, merge_cols=merge_cols)

        # Replace 'viewpoint' values where 'viewpoint_y' is not null
        df.loc[df["viewpoint_y"].notna(), "viewpoint"] = df.loc[df["viewpoint_y"].notna(), "viewpoint_y"]
        df = df.drop("viewpoint_y", axis=1)

    return df

def compute_dataset_statistics(df):
    num_annotations = len(df)
    num_images = len(df['image_uuid'].unique())
    num_individuals = len(df['individual_uuid'].unique())
    num_categories = len(df['category_id'].unique())
    num_individual_viewpoints = len((df['name'] + '-' + df['viewpoint']).unique())

    annotations_per_category = df.groupby('species').size().to_dict()
    annotations_per_viewpoint = df.groupby('viewpoint').size().to_dict()
    images_per_name = df.groupby('name')['image_uuid'].nunique()
    max_images_per_name = images_per_name.max()
    min_images_per_name = images_per_name.min()
    avg_images_per_name = images_per_name.mean()

    print("Dataset Statistics:")
    print(f"Number of annotations: {num_annotations}")
    print(f"Number of images: {num_images}")
    print(f"Number of individuals: {num_individuals}")
    print(f"Number of individual viewpoints: {num_individual_viewpoints}")
    print(f"Number of categories: {num_categories}")

    print("\nAnnotations per Category:")
    for category, count in annotations_per_category.items():
        print(f"{category}: {count}")

    print("\nAnnotations per Viewpoint:")
    for viewpoint, count in annotations_per_viewpoint.items():
        print(f"{viewpoint}: {count}")


    print(f"\nMax images per name: {max_images_per_name}")
    print(f"Min images per name: {min_images_per_name}")
    print(f"Average images per name: {avg_images_per_name:.2f}")

def preprocess(anno_path, images_dir, csv_dir=None, csv_column_names=['annotation_uuid', 'viewpoint_y'], merge_cols=['annotation_uuid', 'viewpoint_y']):
    data = load_json(anno_path)
    dfa = pd.DataFrame(data['annotations'])
    dfi = pd.DataFrame(data['images'])
    if 'categories' in data.keys():
        dfc = pd.DataFrame(data['categories'])
    else:
        species = list(dfa['species'].unique())
        dfc = pd.DataFrame()
        dfc['name'] = species
        dfc['id'] = [i for i in range(0,len(species))]
        dfa['category_id'] = dfa['species'].apply(lambda x: species.index(x))
        
    dfi = dfi.drop_duplicates(keep='first').reset_index(drop=True)

    df = merge_dataframes(dfa, dfi)
    
    df = df.merge(dfc, left_on='category_id', right_on='id', suffixes=('', '_category'))
    df['species'] = df['name_category']
    df['path'] = df['file_name'].apply(lambda x: os.path.join(images_dir, x))

    

    existence_check = df['path'].apply(os.path.exists)
    assert existence_check.all(), f"The following files are not found: {df[~existence_check]['path'].tolist()}"
    
    df = df[existence_check].reset_index(drop=True)

    if csv_dir is not None:
        df = apply_csv_filter(df, csv_dir, csv_column_names, merge_cols)


    df = add_individual_uuid(df)
    
    return df
    

def save_data(df, output_path):
    df_annotations_fields = ['uuid_x', 'image_uuid', 'bbox', 'theta', 'individual_uuid', 'viewpoint', 'category_id']
    df_annotations = df[df_annotations_fields]
    df_annotations = df_annotations.rename(columns={'uuid_x': 'uuid'})


    df_images_fields = ['image_uuid', 'file_name', 'height', 'width', 'date_captured']
    df_images_fields= df.columns.intersection(df_images_fields)
    df_images = df[df_images_fields].drop_duplicates(keep='first').reset_index(drop=True)
    df_images = df_images.rename(columns={'image_uuid': 'uuid'})

    df_individuals_fields = ['individual_uuid', 'name']
    df_individuals = df[df_individuals_fields].drop_duplicates(keep='first').reset_index(drop=True)
    df_individuals = df_individuals.rename(columns={'individual_uuid': 'uuid', 'name': 'name'})

    df_categories_fields = ['category_id', 'species']
    df_categories = df[df_categories_fields].drop_duplicates(keep='first').reset_index(drop=True)
    df_categories = df_categories.rename(columns={'category_id': 'id'})

    result_dict = {'categories': df_categories.to_dict(orient='records'),
                   'images': df_images.to_dict(orient='records'),
                   'annotations': df_annotations.to_dict(orient='records'),
                   'individuals': df_individuals.to_dict(orient='records')}

    with open(output_path, 'w') as f:
        json.dump(result_dict, f, indent=4)
        print('Data is saved to:', output_path)
    return df
    

def main():
    parser = ArgumentParser()
    parser.add_argument(
        "-a",
        "--annotation",
        dest="anno_path",
        required=True,
        help="Path to a file with annotations",
    ),
    parser.add_argument(
        "-s",
        "--source",
        dest="source_dir",
        required=True,
        help="directory containing source images",
    ),
    parser.add_argument(
        "-o",
        "--output",
        dest="output_path",
        required=True,
        help="path to save the json result",
    ),
    parser.add_argument(
        "-c",
        "--csv",
        dest="csv_dir",
        required=False,
        default=None,
        help="directory containing csv files with additional annotations",
    ),
    parser.add_argument(
        "--csv_column_names",
        dest="csv_column_names",
        required=False,
        default="['annotation_uuid', 'viewpoint_y']",
        help="column names in csv files",
    ),
    parser.add_argument(
        "--merge_cols",
        dest="merge_cols",
        required=False,
        default="['annotation_uuid', 'viewpoint_y']",
        help="column names in csv files to be merged",
    ),

    args = parser.parse_args()

    data = preprocess(args.anno_path, args.source_dir, args.csv_dir, args.csv_column_names, args.merge_cols)
    compute_dataset_statistics(data)
    save_data(data, args.output_path)


if __name__ == "__main__":
    main()


