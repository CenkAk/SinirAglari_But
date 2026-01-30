import os
import sys
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from flask import Flask
from app.routes import main
from app.model_loader import ModelLoader

def create_app():
    app = Flask(__name__, static_folder='app/static', static_url_path='/static')
    app.config['SECRET_KEY'] = 'eye-disease-detection-secret-key'
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
    app.config['UPLOAD_FOLDER'] = 'uploads'
    
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    app.register_blueprint(main)
    
    model_path = os.path.join('models', 'best_model.pth')
    mapping_path = os.path.join('models', 'class_mapping.json')

    if os.path.exists(model_path):
        try:
            model_loader = ModelLoader()
            model_loader.load_model(model_path, mapping_path)
            print("Model başarıyla yüklendi!")
        except Exception as e:
            print(f"Model yüklenirken hata: {e}")
            print("Uygulama model olmadan çalışacak. Lütfen önce modeli eğitin.")
    else:
        print("Model dosyası bulunamadı. Lütfen önce modeli eğitin (python train.py)")
    
    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
