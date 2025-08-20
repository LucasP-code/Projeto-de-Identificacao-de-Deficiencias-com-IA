import os
import cv2
import random
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Parâmetros
input_folder = "C:\\Lusca\\WorkSpace\\TG\\CoLeaf DATASET treinamento"
output_folder = "C:\\Lusca\\WorkSpace\\TG\\dataset_processado"
tamanho = (224, 224)
train_ratio = 0.8

# Funções utilitárias
def preprocess_image(img):
    """Redimensiona e normaliza a imagem"""
    img = cv2.resize(img, tamanho)
    img = img / 255.0
    return img

def augment_image(img):
    """Aplica Data Augmentation"""
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.2)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    return img.numpy()

# Criação das pastas de saída
for split in ['treino', 'teste']:
    os.makedirs(os.path.join(output_folder, split), exist_ok=True)

# Loop principal por classe
for class_name in os.listdir(input_folder):
    class_path = os.path.join(input_folder, class_name)
    if not os.path.isdir(class_path):
        continue

    images = []
    for filename in os.listdir(class_path):
        img_path = os.path.join(class_path, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = preprocess_image(img)
            images.append(img)

    # Embaralhar
    random.shuffle(images)

    # Separar treino e teste (80/20)
    train_imgs, test_imgs = train_test_split(images, train_size=train_ratio, random_state=42)

    # Pasta da classe dentro de treino/teste
    for split, split_imgs in [('treino', train_imgs), ('teste', test_imgs)]:
        split_class_dir = os.path.join(output_folder, split, class_name)
        os.makedirs(split_class_dir, exist_ok=True)

        for i, img in enumerate(split_imgs):
            # Aplica Data Augmentation só no treino
            if split == 'treino':
                img = augment_image(img)
            img_uint8 = (img * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(split_class_dir, f"{class_name}_{i}.jpg"), img_uint8)

print("Pré-processamento completo!")
