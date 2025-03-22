import os
import shutil
import random
from pathlib import Path

# Klasör yolları
base_dir = Path(r"C:\Users\melis\OneDrive\Masaüstü\Derin_Ogrenme\Teknofest2\Teknofest21_Inme_Veriseti\Kullanilacak_Veriler\Alt_Kumeler")
out_dir = Path(r"C:\Users\melis\OneDrive\Masaüstü\Derin_Ogrenme\Teknofest2\Teknofest21_Inme_Veriseti\Kullanilacak_Veriler\Alt_Kumeler_Test_Esit")
out_dir.mkdir(parents=True, exist_ok=True)

folds = ["Fold1", "Fold2", "Fold3"]
classes = ["Inme-Yok", "Inme-Var"]

for fold in folds:
    for cls in classes:
        # Kaynak klasörler
        train_dir = base_dir / fold / "train" / cls
        test_dir = base_dir / fold / "test" / cls
        
        # Hedef klasörleri oluştur
        new_train_dir = out_dir / fold / "train" / cls
        new_test_dir = out_dir / fold / "test" / cls
        new_train_dir.mkdir(parents=True, exist_ok=True)
        new_test_dir.mkdir(parents=True, exist_ok=True)
        
        # Dosya listesini al
        train_files = list(train_dir.glob("*.png")) + list(train_dir.glob("*.jpg"))
        test_files = list(test_dir.glob("*.png")) + list(test_dir.glob("*.jpg"))
        
        if cls == "Inme-Var":
            # Eksik dosya sayısını hesapla ve train'den tamamla
            missing = 750 - len(test_files)
            if missing > 0:
                additional_files = random.sample(train_files, missing)
                test_files.extend(additional_files)
                train_files = [f for f in train_files if f not in additional_files]
        else:
            # Fazla dosyaları belirle ve train'e geri taşı
            excess = len(test_files) - 750
            if excess > 0:
                to_move = random.sample(test_files, excess)
                train_files.extend(to_move)
                test_files = [f for f in test_files if f not in to_move]
        
        # Yeni dosyaları kopyala
        for f in train_files:
            shutil.copy(f, new_train_dir / f.name)
        for f in test_files:
            shutil.copy(f, new_test_dir / f.name)

print("İşlem tamamlandı. Dengelenmiş test seti oluşturuldu.")
