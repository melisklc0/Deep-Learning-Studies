import os
import cv2
import numpy as np
from tqdm import tqdm
import random
import shutil

# Klasör yolları
source_path = r"D:\Derin_Ogrenme\Teknofest2\Teknofest21_Inme_Veriseti\Kullanilacak_Veriler\Alt_Kumeler_Test_Esit"
target_path = r"D:\Derin_Ogrenme\Teknofest2\Teknofest21_Inme_Veriseti\Kullanilacak_Veriler\Alt_Kumeler_Augmantasyon"

# Birden fazla augmentasyon uygulama fonksiyonu
def random_augmentation(img):
    h, w = img.shape[:2]
    aug_names = []

    # Rastgele rotasyon
    if random.random() < 0.5:
        angle = random.uniform(-10, 10)
        M_rot = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        img = cv2.warpAffine(img, M_rot, (w, h))
        aug_names.append(f"rot{int(angle)}")

    # Rastgele zoom
    if random.random() < 0.5:
        zoom_factor = random.uniform(1.0, 1.2)
        img_zoom = cv2.resize(img, None, fx=zoom_factor, fy=zoom_factor)
        zh, zw = img_zoom.shape[:2]
        start_x = zh//2 - h//2
        start_y = zw//2 - w//2
        img = img_zoom[start_x:start_x+h, start_y:start_y+w]
        aug_names.append(f"zoom{zoom_factor:.2f}")

    # Rastgele translation
    if random.random() < 0.5:
        tx = random.randint(-10, 10)
        ty = random.randint(-10, 10)
        M_trans = np.float32([[1, 0, tx], [0, 1, ty]])
        img = cv2.warpAffine(img, M_trans, (w, h))
        aug_names.append(f"trans{tx}_{ty}")

    # Rastgele yatay flip
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
        aug_names.append("flip")

    if not aug_names:
        aug_names.append("original")

    aug_name = "_".join(aug_names)
    return img, aug_name

# Ana augmentasyon ve dengeleme fonksiyonu
def augment_and_balance():
    folds = ["Fold1", "Fold2", "Fold3"]
    categories = ["Inme-Yok", "Inme-Var"]
    target_count = 7500

    for fold in folds:
        for category in categories:
            source_dir = os.path.join(source_path, fold, "train", category)
            target_dir = os.path.join(target_path, fold, "train", category)
            os.makedirs(target_dir, exist_ok=True)

            image_files = [f for f in os.listdir(source_dir) if f.endswith((".jpg", ".png", ".jpeg"))]

            # Orijinal görüntüleri hedef klasöre kopyala
            for img_name in tqdm(image_files, desc=f"Kopyalanıyor {fold}/{category}"):
                shutil.copy2(os.path.join(source_dir, img_name), target_dir)

            current_count = len(image_files)
            needed_augments = target_count - current_count

            if needed_augments <= 0:
                print(f"{fold}/{category} zaten {current_count} görüntü içeriyor, atlandı.")
                continue

            print(f"{fold}/{category}: {needed_augments} ek görüntü oluşturulacak.")

            i = 0
            while needed_augments > 0:
                img_name = random.choice(image_files)
                img_path = os.path.join(source_dir, img_name)
                img = cv2.imread(img_path)

                if img is None:
                    print(f"HATA: {img_path} okunamadı! Dosya var mı ve bozuk mu?")
                    continue

                aug_img, aug_desc = random_augmentation(img)

                new_img_name = f"{os.path.splitext(img_name)[0]}{aug_desc}{i}.jpg"
                cv2.imwrite(os.path.join(target_dir, new_img_name), aug_img)

                i += 1
                needed_augments -= 1


if __name__ == "__main__":
    augment_and_balance()

