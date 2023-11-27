# Plot random set of images from dataframe in a grid, with labels

import cv2
import numpy as np
import matplotlib.pyplot as plt
import cv2


def get_image_dimensions(filename):
    img = cv2.imread(filename)
    h, w, c = img.shape
    return h, w

def rotate_box(x1,y1,x2,y2,theta):
    xm = (x1 + x2) // 2
    ym = (y1 + y2) // 2

    h = int(y2 - y1)
    w = int(x2 - x1)

    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    A = np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1], [x1, y1]])
    C = np.array([[xm, ym]])
    RA = (A - C) @ R.T + C
    RA = RA.astype(int)

    return RA

def crop_rect(img, rect):
    center, size, angle = rect[0], rect[1], rect[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))

    height, width = img.shape[0], img.shape[1]
    
    diag_len = int(np.sqrt(height * height + width * width))
    new_width = diag_len
    new_height = diag_len

    blank_canvas = np.ones((new_height, new_width, 3), dtype=img.dtype) * 255

    x_offset = (new_width - width) // 2
    y_offset = (new_height - height) // 2

    blank_canvas[y_offset:y_offset+height, x_offset:x_offset+width] = img

    new_center_x = new_width // 2
    new_center_y = new_height // 2

    M = cv2.getRotationMatrix2D((new_center_x, new_center_y), np.rad2deg(angle), 1)

    img_rot = cv2.warpAffine(blank_canvas, M, (new_width, new_height), flags=cv2.INTER_LINEAR, 
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))

    new_center = np.dot(M[:,:2], np.array([center[0], center[1]]) + np.array([x_offset, y_offset])) + M[:,2]

    img_crop = cv2.getRectSubPix(img_rot, size, new_center)
    return img_crop, img_rot


def get_chip_from_img(img, bbox, theta):
    x1,y1,w,h = bbox
    x2 = x1 + w
    y2 = y1 + h
    xm = (x1 + x2) // 2
    ym = (y1 + y2) // 2

    # Do a faster, regular crop if theta is negligible
    if abs(theta) < 0.1:
        x1, y1, w, h = [int(x) for x in bbox]
        cropped_image = img[y1 : y1 + h, x1 : x1 + w]
    else:
        cropped_image = crop_rect(img, ((xm, ym), (x2-x1, y2-y1), theta))[0]

    if min(cropped_image.shape) < 1:
        # Use original image
        print(f'Using original image. Invalid parameters - theta: {theta}, bbox: {bbox}')
        cropped_image = img

    return cropped_image

def plot_images(df, n_images=24, species=None, viewpoint=None, individual_id=None, large_grid=False, crop_bbox=False):

    if large_grid:
        fig, axes = plt.subplots(nrows=6, ncols=6, figsize=(16, 16))
    else:
        fig, axes = plt.subplots(nrows=3, ncols=6, figsize=(16, 8))

    if species:
        if isinstance(species, str): species = [species]
        df = df[df['species'].isin(species)]
    if individual_id:
        df = df[df['individual_id'] == individual_id]
    if viewpoint:
        df = df[df['viewpoint'] == viewpoint]

    for i, ax in enumerate(axes.flatten()):
        row = df.sample(n=1).iloc[0]
        img_path = row['path']
        theta = row['theta']
        bbox = row['bbox']
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if crop_bbox:
            img = get_chip_from_img(img, bbox, theta)
        ax.imshow(img)
        ax.set_title(row['species'])
        ax.axis('off')
    plt.tight_layout()
    plt.show()

