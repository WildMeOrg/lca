
import os

def set_default(config, data_dir):
    config["data_dir"] = data_dir
    config["export_dir"] =  os.path.join(data_dir, "export_annotations")
    anno_dir = os.path.join(data_dir, "annotations") 
    config["anno_dir"] = anno_dir
    config["train_images_dir"] = os.path.join(data_dir, "images/test2023")
    anno_file = lambda split: f"instances_{split}2023.json"

    config["train_anno_path"] = os.path.join(anno_dir, anno_file("test")) 
    config["val_anno_path"] = os.path.join(anno_dir, anno_file("val")) 
    config["test_anno_path"] = os.path.join(anno_dir, anno_file("train")) 

    config["excluded_viewpoints"] = {'front', 'back'}
    config["csv_folder"] = os.path.join(data_dir, "csvs")
    return config

def config(species):
    config =  {}     
    if species == 'giraffe':
        data_dir = "/media/kate/Elements1/ISYNC-LUT/reticulatedGiraffe/coco" 
    elif species == 'leopard':
        data_dir = "/media/kate/Elements1/ISYNC-LUT/leopard/coco" 
    config = set_default(config, data_dir)
    return config

