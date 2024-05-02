import os
from PIL import Image
from skimage import filters


def apply_median_filter(image_path, save_path):

    # Cargar la imagen
    image = Image.open(image_path)

    # Aplicar el filtro de mediana
    filtered_image = image.filter(filters.median)

    # Guardar la imagen filtrada en la ruta especificada
    filtered_image.save(save_path)

def filter_images_in_folder(folder_path, save_folder):

    # Verificar si la carpeta de guardado existe, si no, crearla
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Obtener la lista de archivos en la carpeta
    image_files = os.listdir(folder_path)

    # Iterar sobre cada archivo en la carpeta
    for file_name in image_files:
        # Comprobar si el archivo es una imagen (extensión .jpg, .png, etc.)
        if file_name.endswith('.jpg') or file_name.endswith('.png'):
            # Construir las rutas de la imagen original y de la imagen filtrada
            image_path = os.path.join(folder_path, file_name)
            save_path = os.path.join(save_folder, file_name)

            # Aplicar el filtro de mediana y guardar la imagen filtrada
            apply_median_filter(image_path, save_path)
# Ejemplo de uso:
folder_path = 'C:/Users/elmer/Downloads/DataTraining2/Data_miguel/coincident_images4'
save_folder = 'C:/Users/elmer/Downloads/DataTraining2/Data_miguel/filtrado'

filter_images_in_folder(folder_path, save_folder)
