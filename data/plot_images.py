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

def plot_images(df, species=None, filter_key="name", filter_value=None, large_grid=False, crop_bbox=False):
    """
    Plot images from a DataFrame with optional filtering and grid size control.

    Parameters:
    - df: DataFrame containing image data.
    - species: List of species names to filter images (default: None for no filtering).
    - filter_key: Name of the column used for additional filtering (default: "name").
    - filter_value: Value to filter the DataFrame by filter_key (default: None for no filtering).
    - large_grid: Boolean to control the size of the grid (default: False).
    - crop_bbox: Boolean to control whether to crop images based on bounding boxes (default: False).

    Returns:
    - None
    """

    # Determine the number of rows and columns for subplots based on grid size
    if large_grid:
        fig, axes = plt.subplots(nrows=6, ncols=6, figsize=(16, 16))
    else:
        fig, axes = plt.subplots(nrows=3, ncols=6, figsize=(16, 8))
    
    # Apply species filter if specified
    if species:
        if isinstance(species, str):
            species = [species]
        df = df[df['species'].isin(species)]
    
    # Apply additional filtering if filter_value is provided
    if filter_value is not None:
        print("Filter value:", filter_value)
        df = df[df[filter_key] == filter_value]
        print("Length:", len(df))
    
    # Shuffle the DataFrame for random image selection
    shuffled_df = df.sample(frac=1).reset_index(drop=True)

    # Iterate through the subplots and display images
    for i, ax in enumerate(axes.flatten()):
        if i >= len(shuffled_df):
            break
        row = shuffled_df.iloc[i]
        img_path = row['path']
        theta = row['theta']
        bbox = row['bbox']
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Optionally crop images based on bounding boxes
        if crop_bbox:
            img = get_chip_from_img(img, bbox, theta)
        
        ax.imshow(img)
        ax.set_title(row['path'].split("/")[-1])
        ax.axis('off')
    
    # Adjust subplot layout and display the plot
    plt.tight_layout()
    plt.show()


def plot_image_pairs(df, species=None, filter_key="name", filter_value=None, crop_bbox=False, N=2):
    """
    Plot images from a DataFrame with optional filtering and grid size control.

    Parameters:
    - df: DataFrame containing image data.
    - species: List of species names to filter images (default: None for no filtering).
    - filter_key: Name of the column used for additional filtering (default: "name").
    - filter_value: Value to filter the DataFrame by filter_key (default: None for no filtering).
    - large_grid: Boolean to control the size of the grid (default: False).
    - crop_bbox: Boolean to control whether to crop images based on bounding boxes (default: False).

    Returns:
    - None
    """

    # Determine the number of rows and columns for subplots based on grid size


    # Adjust nrows and ncols according to your requirements
    nrows = N
    ncols = 2  # One column per filter_value pair in this setup

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 12))

    # Filter DataFrame by species if specified
    if species:
        df = df[df['species'].isin(species)]

    for n in range(N):
    # Sample a unique filter_value
        filter_value = df[filter_key].drop_duplicates().sample().values[0]
        # print(f"Filter value: {filter_value}")
        
        # Filter the DataFrame for rows matching the filter_value
        df_show = df[df[filter_key] == filter_value]
        
        if not df_show.empty:
            
            # We aim to plot only the first two images, so we ensure df_show is limited to at most 2 rows
            df_show = df_show.head(2)
            
            for idx, (_, row) in enumerate(df_show.iterrows()):
                img_path = row['path']
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Optionally crop images based on bounding boxes
                if crop_bbox:
                    theta = row.get('theta', 0)  # Default theta to 0 if not present
                    bbox = row.get('bbox', None)  # Handle missing bbox if necessary
                    if bbox:
                        
                        img = get_chip_from_img(img, bbox, theta)
                # Plotting
                ax = axes[n, idx % 2]  
                ax.imshow(img)
                ax.set_title(f"{filter_value}")
                ax.axis('off')
        else:
            print(f"No images found for filter value: {filter_value}")

    # Adjust subplot layout and display the plot
    plt.tight_layout()
    plt.show()