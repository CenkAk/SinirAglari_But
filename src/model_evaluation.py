import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
import torch
from tqdm import tqdm


class ModelEvaluator:
    def __init__(self, model, class_names, device='cuda'):
        self.model = model
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.device = device

    def generate_all_evaluations(self, test_loader, history=None, save_dir='models'):
        os.makedirs(save_dir, exist_ok=True)
        
        y_pred_proba, y_true = self._get_predictions(test_loader)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        y_test_cat = np.eye(self.num_classes)[y_true]

        metrics = {}
        metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
        report = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            output_dict=True
        )
        metrics['classification_report'] = report
        cm = confusion_matrix(y_true, y_pred)
        self._plot_confusion_matrix(cm, save_dir)
        self._plot_roc_curves(y_test_cat, y_pred_proba, save_dir)
        self._plot_pr_curves(y_test_cat, y_pred_proba, save_dir)
        if history:
            self._plot_training_history(history, save_dir)
        self._save_metrics_csv(metrics, save_dir)

        print(f"\nTest Accuracy: {metrics['accuracy']:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=self.class_names))

        return metrics
    
    def _get_predictions(self, test_loader):
        self.model.eval()
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc='Evaluating'):
                images = images.to(self.device)
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                
                all_probs.append(probs.cpu().numpy())
                all_labels.append(labels.numpy())
        
        y_pred_proba = np.concatenate(all_probs, axis=0)
        y_true = np.concatenate(all_labels, axis=0)
        
        return y_pred_proba, y_true

    def _plot_confusion_matrix(self, cm, save_dir):
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            annot_kws={'size': 8}
        )
        plt.title('Confusion Matrix')
        plt.ylabel('Gerçek')
        plt.xlabel('Tahmin')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=150)
        plt.close()

    def _plot_roc_curves(self, y_test_cat, y_pred_proba, save_dir):
        plt.figure(figsize=(10, 8))
        for i in range(self.num_classes):
            fpr, tpr, _ = roc_curve(y_test_cat[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{self.class_names[i]} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc='lower right', fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'roc_curves.png'), dpi=150)
        plt.close()

    def _plot_pr_curves(self, y_test_cat, y_pred_proba, save_dir):
        plt.figure(figsize=(10, 8))
        for i in range(self.num_classes):
            precision, recall, _ = precision_recall_curve(y_test_cat[:, i], y_pred_proba[:, i])
            ap = average_precision_score(y_test_cat[:, i], y_pred_proba[:, i])
            plt.plot(recall, precision, label=f'{self.class_names[i]} (AP = {ap:.2f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend(loc='lower left', fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'pr_curves.png'), dpi=150)
        plt.close()

    def _plot_training_history(self, history, save_dir):
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        if 'loss' in history:
            axes[0].plot(history['loss'], label='Train Loss')
            axes[0].plot(history['val_loss'], label='Val Loss')
            axes[0].set_title('Loss')
            axes[0].legend()
            axes[0].set_xlabel('Epoch')

        if 'accuracy' in history:
            axes[1].plot(history['accuracy'], label='Train Accuracy')
            axes[1].plot(history['val_accuracy'], label='Val Accuracy')
            axes[1].set_title('Accuracy')
            axes[1].legend()
            axes[1].set_xlabel('Epoch')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=150)
        plt.close()

    def _save_metrics_csv(self, metrics, save_dir):
        import csv
        report = metrics.get('classification_report', {})
        with open(os.path.join(save_dir, 'metrics_table.csv'), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Class', 'Precision', 'Recall', 'F1-Score', 'Support'])
            for name in self.class_names:
                if name in report and isinstance(report[name], dict):
                    row = report[name]
                    writer.writerow([
                        name,
                        f"{row.get('precision', 0):.4f}",
                        f"{row.get('recall', 0):.4f}",
                        f"{row.get('f1-score', 0):.4f}",
                        int(row.get('support', 0))
                    ])
            writer.writerow([])
            writer.writerow(['Accuracy', f"{metrics.get('accuracy', 0):.4f}"])
