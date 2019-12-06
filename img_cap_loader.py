import os
import numpy as np
from keras.preprocessing.image import img_to_array, load_img 


def load_normalized_img_and_its_text(img_dir_path, txt_dir_path, img_width, img_height):

    images = dict()
    texts = dict()
    for f in os.listdir(img_dir_path):
        filepath = os.path.join(img_dir_path, f)
        if os.path.isfile(filepath) and f.endswith('.jpg'):
            name = f.replace('.jpg', '')
            images[name] = filepath
    
    extension = 'txt'
    for dirpath, dirnames, files in os.walk(txt_dir_path):
      for name in files:
        if extension and name.lower().endswith(extension):
          fileName = name.replace('.txt', '')
          texts[fileName] = open(os.path.join(dirpath, name), 'rt').read()

    result = []
    for name, img_path in images.items():
        if name in texts:
            text = texts[name]
            image = img_to_array(load_img(img_path, target_size=(img_width, img_height)))
            image = (image.astype(np.float32) / 255) * 2 - 1
            result.append([image, text])

    return np.array(result)
