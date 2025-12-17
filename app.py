from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pillow_heif
import io

pillow_heif.register_heif_opener()

app = Flask(__name__)

# Load model class
class MobileNetV2Multi(nn.Module):
  def __init__(self, num_ingredients=8, num_freshness=2):
    super().__init__()
    base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    in_feats = base.classifier[1].in_features
    base.classifier = nn.Identity()
    
    self.base = base
    self.ingredient_head = nn.Linear(in_feats, num_ingredients)
    self.freshness_head = nn.Linear(in_feats, num_freshness)

  def forward(self, x):
    feat = self.base(x)
    return self.ingredient_head(feat), self.freshness_head(feat)

# Load trained model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = MobileNetV2Multi(num_ingredients=8, num_freshness=2)
model.load_state_dict(torch.load('mobilenetv2_cv_model.pt', map_location=device))
model.to(device)
model.eval()

print(f"Model loaded successfully on {device.upper()}")

# Labels (sorted alphabetically as in training)
ingredient_names = ['beef', 'chevon', 'egg', 'fish', 'poultry', 'shrimp', 'tempeh', 'tofu']
freshness_names = ['fresh', 'spoiled']

transform = transforms.Compose([
  transforms.Resize((224, 224)),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
  try:
    if 'file' not in request.files:
      return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
      return jsonify({'error': 'No file selected'}), 400
    
    # Read and process image
    img = Image.open(io.BytesIO(file.read())).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
      ing_output, fre_output = model(img_tensor)
      
      # Get predictions
      ing_pred = torch.argmax(ing_output, dim=1).item()
      fre_pred = torch.argmax(fre_output, dim=1).item()
      
      # Get confidence scores
      ing_probs = torch.softmax(ing_output, dim=1)
      fre_probs = torch.softmax(fre_output, dim=1)
      
      ing_confidence = ing_probs[0][ing_pred].item() * 100
      fre_confidence = fre_probs[0][fre_pred].item() * 100
      
    return jsonify({
      'ingredient': ingredient_names[ing_pred],
      'freshness': freshness_names[fre_pred],
      'ingredient_confidence': f"{ing_confidence:.2f}%",
      'freshness_confidence': f"{fre_confidence:.2f}%"
    })
    
  except Exception as e:
    print(f"Error: {str(e)}")
    return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/health')
def health():
  return jsonify({
    'status': 'healthy',
    'model_loaded': True,
    'device': device,
    'ingredients': ingredient_names,
    'freshness_options': freshness_names
  })

if __name__ == '__main__':
  app.run(debug=True, host='0.0.0.0', port=5000)