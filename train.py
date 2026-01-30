import os
import sys
import torch
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_preprocessing import DataPreprocessor
from src.model_training import EyeDiseaseModel, Trainer
from src.model_evaluation import ModelEvaluator
from src.utils import save_class_mapping


def check_gpu():
    print("\nGPU KONTROLÜ")
    print("-"*60)
    
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"✓ CUDA kullanılabilir!")
        print(f"✓ {device_count} GPU cihazı bulundu:")
        for i in range(device_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        print(f"✓ CUDA Version: {torch.version.cuda}")
        print(f"✓ cuDNN Version: {torch.backends.cudnn.version()}")
        
        return 'cuda'
    else:
        print("✗ CUDA kullanılamıyor - CPU kullanılacak")
        print("  (Eğitim süresi daha uzun olabilir)")
        return 'cpu'


def main():
    print("="*60)
    print("GÖZ HASTALIĞI TESPİT SİSTEMİ - MODEL EĞİTİMİ (PyTorch)")
    print("="*60)
    
    device = check_gpu()
    
    IMG_SIZE = (380, 380)
    BATCH_SIZE = 16
    FROZEN_EPOCHS = 3
    FINE_TUNE_EPOCHS = 2
    NUM_WORKERS = 4
    
    print("\n1. VERİ SETİ HAZIRLANIYOR")
    print("-"*60)
    
    preprocessor = DataPreprocessor(img_size=IMG_SIZE)
    
    possible_paths = [
        ('Eye Disease Image Dataset/Eye Disease Image Dataset/Original Dataset/Original Dataset', 
         'Eye Disease Image Dataset/Eye Disease Image Dataset/Augmented Dataset/Augmented Dataset'),
        ('Eye Disease Image Dataset/Eye Disease Image Dataset/Original Dataset', 
         'Eye Disease Image Dataset/Eye Disease Image Dataset/Augmented Dataset'),
        ('data/original', 'data/augmented'),
        ('Original Dataset', 'Augmented Dataset'),
    ]
    
    original_dir = augmented_dir = None
    for orig, aug in possible_paths:
        if os.path.exists(orig) or os.path.exists(aug):
            original_dir = orig if os.path.exists(orig) else None
            augmented_dir = aug if os.path.exists(aug) else None
            break

    try:
        paths_train, paths_val, paths_test, y_train, y_val, y_test, class_weights, class_distribution = \
            preprocessor.prepare_dataset(
                original_dir=original_dir,
                augmented_dir=augmented_dir
            )
    except Exception as e:
        print(f"Hata: Veri seti yüklenemedi: {e}")
        print("\nBeklenen klasör yapısı:")
        print("  Eye Disease Image Dataset/Eye Disease Image Dataset/Original Dataset/Original Dataset/<SınıfAdı>/")
        print("\nSınıf adları: Diabetic Retinopathy, Glaucoma, Healthy, vb.")
        print("\nREADME'deki veri seti talimatlarına bakın.")
        return
    
    save_class_mapping(
        preprocessor.class_to_idx,
        preprocessor.idx_to_class,
        save_path='models/class_mapping.json'
    )
    
    print("\n2. DATA LOADER'LAR OLUŞTURULUYOR")
    print("-"*60)
    
    train_loader, val_loader = preprocessor.get_data_loaders(
        paths_train, y_train, paths_val, y_val, 
        batch_size=BATCH_SIZE, 
        num_workers=NUM_WORKERS
    )
    test_loader = preprocessor.get_test_loader(
        paths_test, y_test, 
        batch_size=BATCH_SIZE, 
        num_workers=NUM_WORKERS
    )
    
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    print("\n3. MODEL OLUŞTURMA VE EĞİTİM")
    print("-"*60)
    print(f"Cihaz: {device.upper()}")
    
    model = EyeDiseaseModel(
        num_classes=len(preprocessor.CLASS_NAMES),
        pretrained=True
    )
    
    trainer = Trainer(model, device=device)
    
    history = trainer.train_two_stage(
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        frozen_epochs=FROZEN_EPOCHS,
        fine_tune_epochs=FINE_TUNE_EPOCHS,
        fine_tune_lr=0.00001,
        save_dir='models'
    )
    
    print("\n4. MODEL DEĞERLENDİRME")
    print("-"*60)
    
    evaluator = ModelEvaluator(model, preprocessor.CLASS_NAMES, device=device)
    metrics = evaluator.generate_all_evaluations(
        test_loader,
        history=history,
        save_dir='models'
    )
    
    print("\n" + "="*60)
    print("EĞİTİM TAMAMLANDI!")
    print("="*60)
    print(f"\nModel dosyası: models/best_model.pth")
    print(f"Değerlendirme görselleri: models/ klasöründe")
    print(f"Eğitim logları: models/training_log.csv")


if __name__ == '__main__':
    main()
