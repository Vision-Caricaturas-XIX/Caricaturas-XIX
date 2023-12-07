import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import cv2
import re
import pandas as pd
import ast


os.environ['PYTHONWARNINGS'] = 'ignore'

def jaccard_index(list1, list2):
    """
    Calculates the Jaccard index between two lists.
    Args:
        list1 (list): First list.
        list2 (list): Second list.
    Returns:
        float: Jaccard index between the two lists. 
    """
    if list1 is np.nan:
        list1 = []
    if list2 is np.nan:
        list2 = []
    intersection = len(set(list1).intersection(set(list2)))
    union = len(set(list1).union(set(list2)))
    return intersection / union if union != 0 else np.nan



def display_images_in_rows(original_paths, result_images, original_titles, result_titles, save_path=None, gray=False):
    """
    Function to plot and compare images horizontally in a matplotlib grid.
    Args:
    - original_paths (list): List of paths to the original images.
    - result_images (list): List of paths to the result images.
    - original_titles (list): List of titles for the original images.
    - result_titles (list): List of titles for the result images.
    - save_path (str): Path to save the figure as an image.
    """
    # Number of rows
    n_rows = len(original_paths)

    # Create a figure with n_rows x 2 subplots
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 6 * n_rows))
    
    for i, (original_path, cropped_image) in enumerate(zip(original_paths, result_images)):
        # Display the original image on the left subplot
        axes[i, 0].imshow(Image.open(original_path))
        axes[i, 0].set_title(original_titles)

        # Display the cropped image on the right subplot
        if gray:
            axes[i, 1].imshow(Image.open(cropped_image), cmap='gray')
        else:
            axes[i, 1].imshow(Image.open(cropped_image))
        axes[i, 1].set_title(result_titles)

    # Display the figure
    plt.tight_layout()

    # Save the figure as an image if save_path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        


def plot_image_comparison_with_zoom(images, directories, zoom_proportions,  titles):
    """
    Function to plot and compare images horizontally in a matplotlib grid, with an additional column showing zoomed-in versions.
    Args:
    - images (list): List of image names to plot.
    - directories (list): List of directories containing the images. Each directory represents a different version of the image.
    - zoom_proportions (list): List of zoom proportions for each directory's image.
    """
    num_dirs = len(directories)
    if num_dirs < 2 or num_dirs > 4:
        raise ValueError("Number of directories should be between 2 and 4.")

    for image_name in images:
        # Load images from each directory and their zoomed versions
        loaded_images = []
        zoomed_images = []
        for dir, zoom in zip(directories, zoom_proportions):
            filename = os.path.join(dir, f"{image_name}_cropped.jpg")
            loaded_image = Image.open(filename)

            # Calculating the crop box for zoom
            width, height = loaded_image.size
            zoom_factor = 1 / zoom
            x_center, y_center = width / 2, height / 2
            box = (x_center - width * zoom_factor / 2, y_center - height * zoom_factor / 2,
                   x_center + width * zoom_factor / 2, y_center + height * zoom_factor / 2)

            zoomed_image = loaded_image.crop(box)
            loaded_images.append(loaded_image)
            zoomed_images.append(zoomed_image)

        # Create a figure and a set of subplots
        fig, axes = plt.subplots(1, num_dirs * 2, figsize=(5 * num_dirs * 2, 5))

        # Plot each image and its zoomed version in respective subplots
        for i in range(num_dirs):
            axes[i].imshow(loaded_images[i])
            axes[i].set_title(f'Image  {titles[i]}')
            axes[i].axis('off')

            axes[i + num_dirs].imshow(zoomed_images[i])
            axes[i + num_dirs].set_title(f'Zoom {zoom_proportions[i]}x {titles[i]}')
            axes[i + num_dirs].axis('off')

        plt.show()


def extraer_palabras_unicas(lista_strings):
    """
    Recibe una lista de strings y devuelve una lista de palabras únicas.
    
    Args:
        lista_strings (list): Lista de strings.
    Returns:
        list: Lista de palabras únicas.
    """ 
    
    palabras_unicas = set()
    for string in lista_strings:
        string_limpio = re.sub(r'\W+', ' ', string.lower())
        palabras = string_limpio.split()
        palabras_unicas.update(palabras)
    return list(palabras_unicas)


def prepare_labels(df_labels):
    """
    Prepara los labels para ser utilizados en el modelo de detección de objetos.
    
    Args:
        df_labels (pd.DataFrame): DataFrame con los labels.
    Returns:
        df_true_text (pd.DataFrame): DataFrame con el id de la imagen y el texto único.
        df_true_objects (pd.DataFrame): DataFrame con el id de la imagen y los objetos únicos.
    """

    df_labels['true_text'] = df_labels['ocr_text'].apply(ast.literal_eval)
    df_labels['true_text'] = df_labels['true_text'].apply(extraer_palabras_unicas)
    df_labels['true_objects'] = df_labels['Labels'].apply(ast.literal_eval)
    df_labels = df_labels.rename(columns={"Id_Imagen":"id"})
    df_labels = df_labels[["id", "true_text", "true_objects"]]
    df_true_text = df_labels[["id", "true_text"]]
    df_true_objects = df_labels[["id", "true_objects"]]
    df_true_objects['true_objects'] = df_true_objects['true_objects'].apply(lambda x: ["woman" if obj=="women" else "man" if obj=="men" else obj for obj in x])

    return df_true_text, df_true_objects
