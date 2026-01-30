import json
import os


def save_class_mapping(class_to_idx, idx_to_class, save_path='models/class_mapping.json'):
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    mapping = {
        'class_to_idx': {k: int(v) for k, v in class_to_idx.items()},
        'idx_to_class': {str(k): v for k, v in idx_to_class.items()}
    }
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
