import os
import shutil
import numpy as np
from sklearn.model_selection import StratifiedKFold

# Veri setinin yolu
data_dir = r"C:\Users\melis\OneDrive\Masaüstü\Derin Öğrenme\Teknofest\Teknofest21 İnme Veriseti\Kullanılacak Veriler"
output_dir = r"C:\Users\melis\OneDrive\Masaüstü\Derin Öğrenme\Teknofest\Teknofest21 İnme Veriseti\Kullanılacak Veriler\Alt_Kümeler"

# Sınıfların listesi
classes = ["Inme-Yok", "Inme-Var"]

# Dosyaları ve etiketleri toplayalım
all_files = []
labels = []

for cls in classes:
    cls_path = os.path.join(data_dir, cls)
    files = os.listdir(cls_path)

    all_files.extend([(cls, file) for file in files])  # (Sınıf, Dosya Adı)
    labels.extend([cls] * len(files))  # Etiketler

# StratifiedKFold ile 3 parçaya bölme
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Fold'ları oluştur
for fold_idx, (train_idx, test_idx) in enumerate(skf.split(all_files, labels)):
    print(f"Fold {fold_idx + 1} oluşturuluyor...")

    fold_path = os.path.join(output_dir, f"Fold{fold_idx + 1}")

    for dataset, idx_set, subset in [("train", train_idx, "train"), ("test", test_idx, "test")]:
        for i in idx_set:
            cls, file = all_files[i]
            src = os.path.join(data_dir, cls, file)
            dst = os.path.join(fold_path, subset, cls, file)

            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(src, dst)

    print(f"Fold {fold_idx + 1} tamamlandı!")

print("Tüm Fold'lar başarıyla oluşturuldu!")
