import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

CLASS_NAMES = [
    'Retinitis Pigmentosa',
    'Retina Dekolmanı',
    'Pterjium',
    'Miyopi',
    'Maküler Skar',
    'Glokom',
    'Disk Ödemesi',
    'Diyabetik Retinopati',
    'Santral Seröz Korioretinopati',
    'Sağlıklı'
]

DATASET_FOLDER_TO_IDX = {
    'Retinitis Pigmentosa': 0,
    'Retinal Detachment': 1,
    'Pterygium': 2,
    'Myopia': 3,
    'Macular Scar': 4,
    'Glaucoma': 5,
    'Disc Edema': 6,
    'Diabetic Retinopathy': 7,
    'Central Serous Chorioretinopathy [Color Fundus]': 8,
    'Healthy': 9,
}


class EyeDiseaseDataset(Dataset):
    def __init__(self, image_paths, labels, img_size=(380, 380), transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.img_size = img_size
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            image = image.resize(self.img_size)
        except Exception as e:
            print(f"Hata: {img_path} - {e}")
            image = Image.new('RGB', self.img_size, (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        
        return image, torch.tensor(label, dtype=torch.long)


class DataPreprocessor:
    CLASS_NAMES = CLASS_NAMES

    def __init__(self, img_size=(380, 380)):
        self.img_size = img_size
        self.class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}

    def _collect_image_paths(self, base_dir):
        image_paths = []
        labels = []

        if not os.path.exists(base_dir):
            return [], []

        for folder_name in os.listdir(base_dir):
            class_path = os.path.join(base_dir, folder_name)
            if not os.path.isdir(class_path):
                continue

            class_idx = DATASET_FOLDER_TO_IDX.get(folder_name)
            if class_idx is None:
                class_idx = self.class_to_idx.get(folder_name)
            if class_idx is None:
                continue

            for filename in os.listdir(class_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    img_path = os.path.join(class_path, filename)
                    image_paths.append(img_path)
                    labels.append(class_idx)

        return image_paths, labels

    def prepare_dataset(self, original_dir=None, augmented_dir=None):
        all_paths = []
        all_labels = []

        dirs_to_load = []
        if augmented_dir and os.path.exists(augmented_dir):
            dirs_to_load.append(augmented_dir)
        if original_dir and os.path.exists(original_dir):
            dirs_to_load.append(original_dir)

        if not dirs_to_load:
            raise ValueError(
                "Veri seti bulunamadı. Lütfen 'Eye Disease Image Dataset' klasörünü kontrol edin. "
                "Original Dataset veya Augmented Dataset klasörleri gerekli."
            )

        for base_dir in dirs_to_load:
            print(f"  Taranıyor: {base_dir}")
            paths, lbls = self._collect_image_paths(base_dir)
            if paths:
                all_paths.extend(paths)
                all_labels.extend(lbls)
                print(f"    {len(paths)} görüntü bulundu")

        if not all_paths:
            raise ValueError("Hiç görüntü bulunamadı. Klasör yapısını kontrol edin.")

        paths = np.array(all_paths)
        labels = np.array(all_labels)

        unique, counts = np.unique(labels, return_counts=True)
        class_distribution = dict(zip([self.idx_to_class[u] for u in unique], counts.tolist()))

        paths_train, paths_temp, y_train, y_temp = train_test_split(
            paths, labels, test_size=0.3, stratify=labels, random_state=42
        )
        paths_val, paths_test, y_val, y_test = train_test_split(
            paths_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
        )

        classes = np.unique(y_train)
        class_weights_array = compute_class_weight(
            'balanced', classes=classes, y=y_train
        )
        class_weights = torch.tensor(class_weights_array, dtype=torch.float32)

        print(f"\nVeri seti hazırlandı: {len(paths)} görüntü")
        print(f"  Train: {len(paths_train)}, Val: {len(paths_val)}, Test: {len(paths_test)}")
        print(f"  Sınıf dağılımı: {class_distribution}")

        return paths_train, paths_val, paths_test, y_train, y_val, y_test, class_weights, class_distribution

    def get_data_loaders(self, paths_train, y_train, paths_val, y_val, batch_size=32, num_workers=4):
        train_transform = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        train_dataset = EyeDiseaseDataset(
            paths_train, y_train, 
            img_size=self.img_size, 
            transform=train_transform
        )
        val_dataset = EyeDiseaseDataset(
            paths_val, y_val, 
            img_size=self.img_size, 
            transform=val_transform
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False
        )
        
        return train_loader, val_loader
    
    def get_test_loader(self, paths_test, y_test, batch_size=32, num_workers=4):
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        test_dataset = EyeDiseaseDataset(
            paths_test, y_test, 
            img_size=self.img_size, 
            transform=test_transform
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True
        )
        
        return test_loader
