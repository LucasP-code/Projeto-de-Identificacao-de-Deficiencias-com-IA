import os
import cv2
import numpy as np

# Diretórios de entrada e saída
input_dir = "dataset"
output_dir = "cropped_dataset"
os.makedirs(output_dir, exist_ok=True)

# Lista de pastas do dataset
folders = ["boron-B", "calcium-Ca", "healthy", 
           "iron-Fe", "magnesium-Mg", "manganese-Mn",
           "nitrogen-N", "phosphorus-P", "potasium-K"]

def adaptive_crop_leaf(image_path):
    """Usa um filtro adaptativo para remover fundo branco e recortar a folha."""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Aplicar desfoque para reduzir ruídos
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Criar uma máscara com limiar adaptativo
    mask = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Remover ruídos pequenos com operações morfológicas
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Encontrar contornos da folha
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Selecionar o maior contorno (provavelmente a folha)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cropped_image = image[y:y+h, x:x+w]
        return cropped_image

    return image  # Retorna a original se não encontrar contornos

# Processar todas as imagens no dataset
for folder in folders:
    input_folder = os.path.join(input_dir, folder)
    output_folder = os.path.join(output_dir, folder)
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            cropped_image = adaptive_crop_leaf(image_path)

            # Salvar a imagem recortada
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, cropped_image)

print("✅ Processamento concluído! As folhas recortadas estão em 'cropped_dataset'.")
