from tools import *
from preprocess import load_to_df
import numpy as np
from PIL import Image

from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet
from lightglue.utils import load_image, rbd, numpy_image_to_torch
from lightglue import viz2d
import torch    
from scipy import integrate
from tqdm.auto import tqdm

from stopwatch import stopwatch

torch.set_grad_enabled(False)


config_path = 'configs/config_whaleshark.yaml'


def crop_to_bbox(img, bbox):
    """Helper function to crop image to bounding box"""
    x, y, w, h = bbox
    img_width, img_height = img.size
    
    # Convert relative coordinates to absolute if needed
    if x <= 1 and y <= 1:  # If coordinates are relative
        x = int(x * img_width)
        y = int(y * img_height)
        w = int(w * img_width)
        h = int(h * img_height)
    else:
        x, y, w, h = map(int, [x, y, w, h])
    
    # Ensure coordinates are within image bounds
    x = max(0, x)
    y = max(0, y)
    w = min(w, img_width - x)
    h = min(h, img_height - y)
    
    # Crop image using x, y, width, height format
    cropped_img = img.crop((x, y, x + w, y + h))
    return cropped_img


def pil_to_torch(image):
    return numpy_image_to_torch(np.array(image))

def estimate_image_similarity_from_matches(match_scores, normalize=True):
    """
    Estimate image similarity using area under the curve of feature match scores.
    
    Parameters:
    -----------
    match_scores : np.ndarray
        Array of match confidence/score values from LightGlue
    normalize : bool, optional
        If True, normalize the AUC to be between 0 and 1 (default is True)
    
    Returns:
    --------
    float
        Similarity score based on the area under the curve of match scores
    """
    # Sort match scores in descending order
    sorted_scores = np.sort(match_scores)[::-1]
    
    # Create x-axis (normalized ranks)
    x = np.linspace(0, 1, len(sorted_scores))
    
    # Compute area under the curve
    auc = integrate.trapz(sorted_scores, x)
    
    # Normalize AUC if requested
    if normalize:
        # Normalize to [0, 1] range
        max_possible_auc = integrate.trapz(np.ones_like(sorted_scores), x)
        if max_possible_auc > 0:
            auc = auc / max_possible_auc
    
    return auc

# def get_lightglue_score(image0, image1, extractor, matcher, device):
#     feats0 = extractor.extract(image0.to(device))
#     feats1 = extractor.extract(image1.to(device))
#     matches01 = rbd(matcher({"image0": feats0, "image1": feats1}))
#     return estimate_image_similarity_from_matches(matches01["scores"].cpu(), normalize=True)
def get_lightglue_score(feats0, feats1, matcher):
    matches01 = rbd(matcher({"image0": feats0, "image1": feats1}))
    return estimate_image_similarity_from_matches(matches01["scores"].cpu(), normalize=True)


def max_resize(img, max_side_size=512):
    if np.prod(img.size) == 0:
        return img
    ratio = max_side_size/max(*img.size)
    new_sz = (int(x * ratio) for x in img.size)
    if np.prod(new_sz) == 0:
        return img
    return img.resize(new_sz)


def process_image(path, bbox, extractor, device):
    img = Image.open(path)
    if bbox is not None:
        img = crop_to_bbox(img, bbox)
    # if img is not None:
    #     img = max_resize(img)
    if img is not None and np.prod(img.size) > 0:
        feats = extractor.extract(pil_to_torch(img).to(device))
    else:
        feats = None
    return feats
    

def calculate_lightglue_scores(uuid_to_imagepath, uuid_to_bbox, extractor, matcher, device):
    vals = list(uuid_to_imagepath.items())

    # vals = vals[:10]
    
    uuids = [x[0] for x in vals]
    print(f"Extracting features...")
    features = [process_image(path, uuid_to_bbox.get(uuid), extractor, device) for (uuid, path) in tqdm(vals)]
    print(f"Extracted features")

    n = len(vals)
    scores = np.zeros((n, n))
    for (i, feats1) in tqdm(list(enumerate(features))):
        for (j, feats2) in tqdm(list(enumerate(features[i+1:]))):
            score = 0
            # print(f"Img 1 {img1.size}, img 2 {img2.size}")
            if (feats1 is None) or (feats2 is None):
                score = 0
            else:
                score = get_lightglue_score(feats1, feats2, matcher)
            
            scores[i, j+i+1] = score
            scores[j+i+1, i] = score
    return {"scores": scores, "uuids": uuids}

def main(config_path):
    config = get_config(config_path)
    annotation_file = config['data']['annotation_file']
    images_dir = config['data']['images_dir']

    data_params = config['data']
    species = config['species']

    embeddings, uuids = load_pickle(data_params['embedding_file'])

    df = load_to_df(annotation_file, format='old')

    df['image_path'] = images_dir + '/' + df['file_name']


    uuid_to_imagepath = df.set_index('uuid_x')['image_path'].to_dict()
    uuid_to_name = df.set_index('uuid_x')['name_viewpoint'].to_dict()  
    uuid_to_bbox = df.set_index('uuid_x')['bbox'].to_dict()  

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'

    extractors = {
        "superpoint":SuperPoint(max_num_keypoints=2048).eval().to(device),
        "disk":DISK(max_num_keypoints=2048).eval().to(device),
        "aliked":ALIKED(max_num_keypoints=2048).eval().to(device),
        "doghardnet":DoGHardNet(max_num_keypoints=2048).eval().to(device)
        }

    for extractor_name, extractor in extractors.items():
        print(f"Using {extractor_name} extractor")
        matcher = LightGlue(features=extractor_name).eval().to(device)

        scores = calculate_lightglue_scores(uuid_to_imagepath, uuid_to_bbox, extractor, matcher, device)

        save_pickle(scores, f"lightglue_scores_{extractor_name}.pickle")


if __name__=="__main__":
    main(config_path)