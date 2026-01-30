import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models
from tqdm import tqdm
import csv


class EyeDiseaseModel(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super(EyeDiseaseModel, self).__init__()
        
        self.backbone = models.efficientnet_b3(
            weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        )
        
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )
        
        self.num_classes = num_classes
    
    def forward(self, x):
        return self.backbone(x)
    
    def freeze_backbone(self):
        for param in self.backbone.features.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self, unfreeze_from=-100):
        for param in self.backbone.features.parameters():
            param.requires_grad = True
        
        layers = list(self.backbone.features.children())
        for layer in layers[:max(0, len(layers) + unfreeze_from)]:
            for param in layer.parameters():
                param.requires_grad = False


class Trainer:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.history = {
            'loss': [], 'accuracy': [],
            'val_loss': [], 'val_accuracy': []
        }
    
    def train_epoch(self, train_loader, criterion, optimizer):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training', leave=False)
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader, criterion):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc
    
    def train_two_stage(
        self,
        train_loader,
        val_loader,
        class_weights=None,
        frozen_epochs=10,
        fine_tune_epochs=50,
        fine_tune_lr=0.00001,
        save_dir='models'
    ):
        os.makedirs(save_dir, exist_ok=True)
        
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0.0
        best_model_wts = copy.deepcopy(self.model.state_dict())
        patience_counter = 0
        patience = 10
        
        csv_file = open(os.path.join(save_dir, 'training_log.csv'), 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['epoch', 'loss', 'accuracy', 'val_loss', 'val_accuracy', 'lr'])
        
        print("\n--- Aşama 1: Frozen base model eğitimi ---")
        self.model.freeze_backbone()
        
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=0.0001
        )
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7)
        
        for epoch in range(frozen_epochs):
            print(f"\nEpoch {epoch+1}/{frozen_epochs}")
            
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            self.history['loss'].append(train_loss)
            self.history['accuracy'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_acc)
            
            csv_writer.writerow([
                len(self.history['loss']), 
                f'{train_loss:.4f}', f'{train_acc:.4f}',
                f'{val_loss:.4f}', f'{val_acc:.4f}',
                f'{current_lr:.2e}'
            ])
            csv_file.flush()
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, LR: {current_lr:.2e}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_wts = copy.deepcopy(self.model.state_dict())
                torch.save(self.model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
                print(f"✓ En iyi model kaydedildi (Val Acc: {val_acc:.4f})")
                patience_counter = 0
            else:
                patience_counter += 1
        
        print("\n--- Aşama 2: Fine-tuning ---")
        self.model.unfreeze_backbone(unfreeze_from=-100)
        
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=fine_tune_lr
        )
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7)
        patience_counter = 0
        
        for epoch in range(fine_tune_epochs):
            print(f"\nEpoch {epoch+1}/{fine_tune_epochs} (Fine-tune)")
            
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            self.history['loss'].append(train_loss)
            self.history['accuracy'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_acc)
            
            csv_writer.writerow([
                len(self.history['loss']),
                f'{train_loss:.4f}', f'{train_acc:.4f}',
                f'{val_loss:.4f}', f'{val_acc:.4f}',
                f'{current_lr:.2e}'
            ])
            csv_file.flush()
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, LR: {current_lr:.2e}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_wts = copy.deepcopy(self.model.state_dict())
                torch.save(self.model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
                print(f"✓ En iyi model kaydedildi (Val Acc: {val_acc:.4f})")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping! {patience} epoch boyunca iyileşme yok.")
                    break
        
        csv_file.close()
        
        self.model.load_state_dict(best_model_wts)
        print(f"\nEn iyi model yüklendi (Val Acc: {best_val_acc:.4f})")
        
        return self.history
