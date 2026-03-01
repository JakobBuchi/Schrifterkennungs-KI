import os
import cv2
import numpy as np

# Vollständiger Pfad zu deinen Trainingsdaten
images_folder = r'C:\Users\jakob\OneDrive\Desktop\HTL Schule\2022_23\KISY\Buchstaben_Finden\Trainingsdaten'

IMG_SIZE = 32  # Quadratisch, 25-50 Pixel

# Unterordner (Klassen) sortiert: A=0, B=1, ..., Z=25
class_folders = sorted([d for d in os.listdir(images_folder) 
                        if os.path.isdir(os.path.join(images_folder, d)) and len(d) == 1 and d.isalpha()])

print(f"Gefundene Klassen (Labels 0-{len(class_folders)-1}): {class_folders}")

images_list = []
labels_list = []

for label_idx, class_folder in enumerate(class_folders):
    class_path = os.path.join(images_folder, class_folder)
    
    image_files = [f for f in os.listdir(class_path) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    print(f"Verarbeite {len(image_files)} Bilder aus {class_folder} (Label {label_idx})")
    
    for filename in image_files:
        filepath = os.path.join(class_path, filename)
        
        image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Fehler beim Laden von {filepath}")
            continue
        
        resized_img = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        normalized_img = resized_img.astype(np.float32) / 255.0  # 0-1 Bereich
        
        images_list.append(normalized_img)
        labels_list.append(label_idx)

X = np.array(images_list)  # (N, 32, 32)
y = np.array(labels_list).reshape(-1, 1)  # (N, 1)

print(f"X Shape: {X.shape}")
print(f"y Shape: {y.shape}")
print(f"X-Wertebereich: [{X.min():.3f}, {X.max():.3f}]")
print(f"Klassenverteilung: {np.bincount(y.flatten())}")  # Neu: Balance checken!

np.save('train_images.npy', X)
np.save('train_labels.npy', y)

print("Fertig! Dateien: train_images.npy und train_labels.npy")
