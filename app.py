from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pillow_heif
import io
import kagglehub
import pandas as pd
import os
from googletrans import Translator

translator = Translator()

def translate_id_to_en(text):
    if not text or pd.isna(text):
        return text
    try:
        return translator.translate(text, src='id', dest='en').text
    except Exception as e:
        print("Translation error:", e)
        return text

pillow_heif.register_heif_opener()

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Download and load Indonesian recipes dataset
print("Downloading Indonesian recipes dataset...")
dataset_path = kagglehub.dataset_download("albertnathaniel12/food-recipes-dataset")
print(f"Path to dataset files: {dataset_path}")

# Load the CSV file
csv_files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
if csv_files:
    recipes_df = pd.read_csv(os.path.join(dataset_path, csv_files[0]))
    # recipes_df.to_csv('recipes_loaded.csv')
    print(f"Loaded {len(recipes_df)} recipes from dataset")
else:
    print("Warning: No CSV file found in dataset")
    recipes_df = pd.DataFrame()

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

# Load your trained model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = MobileNetV2Multi(num_ingredients=8, num_freshness=2)
model.load_state_dict(torch.load('mobilenetv2_cv_model.pt', map_location=device))
model.to(device)
model.eval()

print(f"Model loaded successfully on {device.upper()}")

# Labels (sorted alphabetically as in training)
ingredient_names = ['beef', 'chevon', 'egg', 'fish', 'poultry', 'shrimp', 'tempeh', 'tofu']
freshness_names = ['Fresh', 'Spoiled']

# Transform - MUST match training transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Mapping ingredients to search keywords
INGREDIENT_KEYWORDS = {
    'beef': ['beef', 'daging sapi', 'sapi'],
    'poultry': ['chicken', 'ayam'],
    'fish': ['fish', 'ikan'],
    'shrimp': ['shrimp', 'udang', 'prawn'],
    'egg': ['egg', 'telur'],
    'tofu': ['tofu', 'tahu'],
    'tempeh': ['tempeh', 'tempe'],
    'chevon': ['goat', 'kambing', 'lamb', 'chevon']
}

def get_recipes_for_ingredient(ingredient, max_recipes=3):
    if recipes_df.empty:
        return []

    keywords = INGREDIENT_KEYWORDS.get(ingredient.lower(), [ingredient.lower()])

    # 1. Filter matching recipes first
    mask = recipes_df['Title'].str.lower().fillna('').apply(
        lambda title: any(k in title for k in keywords)
    )

    matched_df = recipes_df[mask]

    if matched_df.empty:
        return []

    # 2. Randomly sample different recipes each time
    sampled_df = matched_df.sample(
        n=min(max_recipes, len(matched_df))
    )

    results = []

    for _, row in sampled_df.iterrows():
        ingredients_en = []
        instructions_en = []

        if pd.notna(row.get('Ingredients')):
            ingredients_en = [
                translate_id_to_en(ing.strip())
                for ing in str(row['Ingredients']).split(',')
            ]

        if pd.notna(row.get('Steps')):
            instructions_en = [
                translate_id_to_en(step.strip())
                for step in str(row['Steps']).split('.')
                if step.strip()
            ]

        results.append({
            'name': row.get('Title', 'Unknown Recipe'),
            'name_translation': translate_id_to_en(row.get('Title')),
            'ingredients': ingredients_en,
            'instructions': instructions_en
        })

    return results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Check if image is in request
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
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
        
        ingredient = ingredient_names[ing_pred]
        quality = freshness_names[fre_pred]
        
        # Get recipes from Kaggle dataset only if fresh
        recipes = get_recipes_for_ingredient(ingredient, max_recipes=3) if quality == 'Fresh' else []
        
        return jsonify({
            'success': True,
            'ingredient': ingredient.capitalize(),
            'quality': quality,
            'confidence': f"{fre_confidence:.1f}",
            'ingredient_confidence': f"{ing_confidence:.1f}",
            'recipes': recipes
        })
    
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'device': device,
        'ingredients': ingredient_names,
        'freshness_options': freshness_names,
        'recipes_loaded': len(recipes_df) if not recipes_df.empty else 0
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)