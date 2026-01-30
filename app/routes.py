import os
from flask import Blueprint, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import io
from app.model_loader import ModelLoader

main = Blueprint('main', __name__)

model_loader = ModelLoader()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@main.route('/')
def index():
    return render_template('index.html')


@main.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Dosya bulunamadı'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'Dosya seçilmedi'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Geçersiz dosya formatı. PNG, JPG, JPEG desteklenir.'}), 400
        
        try:
            image = Image.open(io.BytesIO(file.read()))
            image = image.convert('RGB')
        except Exception as e:
            return jsonify({'error': f'Görüntü yüklenemedi: {str(e)}'}), 400
        
        if model_loader.model is None:
            model_path = os.path.join('models', 'best_model.pth')
            mapping_path = os.path.join('models', 'class_mapping.json')
            
            if not os.path.exists(model_path):
                return jsonify({
                    'error': 'Model dosyası bulunamadı. Lütfen önce modeli eğitin.'
                }), 500
            
            model_loader.load_model(model_path, mapping_path)
        
        predictions = model_loader.predict(image)
        
        sorted_predictions = sorted(
            predictions['predictions'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return jsonify({
            'success': True,
            'top_class': predictions['top_class'],
            'top_probability': round(predictions['top_probability'] * 100, 2),
            'predictions': [
                {'class': cls, 'probability': round(prob * 100, 2)}
                for cls, prob in sorted_predictions
            ],
            'all_classes': predictions['all_classes']
        })
    
    except Exception as e:
        return jsonify({'error': f'Bir hata oluştu: {str(e)}'}), 500


@main.route('/health')
def health():
    model_loaded = model_loader.model is not None
    return jsonify({
        'status': 'ok',
        'model_loaded': model_loaded
    })
