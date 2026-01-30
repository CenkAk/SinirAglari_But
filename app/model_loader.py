import os
import torch
import json
import numpy as np
from PIL import Image
from torchvision import transforms

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model_training import EyeDiseaseModel


class ModelLoader:
    _instance = None
    _model = None
    _class_names = None
    _img_size = (380, 380)
    _device = None
    _transform = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance
    
    def load_model(self, model_path='models/best_model.pth', mapping_path='models/class_mapping.json'):
        if self._model is None:
            try:
                self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                print(f"Cihaz: {self._device}")
                if os.path.exists(mapping_path):
                    with open(mapping_path, 'r', encoding='utf-8') as f:
                        mapping = json.load(f)
                    self._class_names = [
                        mapping['idx_to_class'][str(i)] 
                        for i in range(len(mapping['idx_to_class']))
                    ]
                else:
                    self._class_names = [
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
                print(f"Model yükleniyor: {model_path}")
                self._model = EyeDiseaseModel(
                    num_classes=len(self._class_names),
                    pretrained=False
                )
                self._model.load_state_dict(torch.load(model_path, map_location=self._device))
                self._model.to(self._device)
                self._model.eval()
                print("Model başarıyla yüklendi!")
                self._transform = transforms.Compose([
                    transforms.Resize(self._img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
            except Exception as e:
                print(f"Model yüklenirken hata: {e}")
                raise
    
    def preprocess_image(self, image_file):
        if not isinstance(image_file, Image.Image):
            img = Image.open(image_file).convert('RGB')
        else:
            img = image_file.convert('RGB')
        
        img_tensor = self._transform(img)
        img_tensor = img_tensor.unsqueeze(0)
        return img_tensor
    
    def predict(self, image_file):
        if self._model is None:
            raise ValueError("Model yüklenmemiş! Önce load_model() çağırın.")
        
        img_tensor = self.preprocess_image(image_file)
        img_tensor = img_tensor.to(self._device)
        
        with torch.no_grad():
            outputs = self._model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
        
        predictions = probabilities.cpu().numpy()
        
        results = {}
        for idx, prob in enumerate(predictions):
            results[self._class_names[idx]] = float(prob)
        
        top_idx = np.argmax(predictions)
        top_class = self._class_names[top_idx]
        top_prob = float(predictions[top_idx])
        
        return {
            'predictions': results,
            'top_class': top_class,
            'top_probability': top_prob,
            'all_classes': self._class_names
        }
    
    @property
    def model(self):
        return self._model
    
    @property
    def class_names(self):
        return self._class_names
