# Ingreadyent
**Image-Based Ingredient Identification with Freshness Detection (MobileNetV2 + Flask)**

Ingreadyent is a lightweight, mobile-friendly web application that identifies food ingredients from images and predicts their quality (Fresh / Spoiled).  
Primarily, it is dedicated to exchange students, travelers, and beginner cooks who might struggle to identify ingredients visually or due to language barriers.

---

## Features
- Upload an ingredient image
- Predict **ingredient type** (8 food)
- Predict **state** (Fresh or Spoiled)
- Fast inference using a lightweight MobileNetV2 model
- Web-based interface powered by Flask

---

## Supported Classes

### Ingredient Categories
- Beef  
- Chevon  
- Egg  
- Fish  
- Poultry  
- Shrimp  
- Tempeh  
- Tofu  

### Freshness Categories
- Fresh  
- Spoiled  

---

## Tech Stack
- **Backend**: Python, Flask
- **Model**: PyTorch, MobileNetV2 (transfer learning)
- **Frontend**: HTML, CSS, JavaScript (Flask templates)
- **Deployment**: GitHub Codespaces or local machine

---

## Project Structure
```text
.
├── app.py                      # Flask application (routes & inference)
├── templates/                  # HTML templates for UI
├── mobilenetv2_model.py        # Model architecture & loading logic
├── mobilenetv2_cv_model.pt     # Trained PyTorch model weights
├── ingredients-dataset/        # Dataset (for training)
├── requirements.txt            # Python dependencies
├── README.md
└── .gitignore
```

---

## Run on GitHub Codespaces (Recommended)

### Step 1: Open in Codespaces
1. Open this repository on GitHub
2. Click **Code** → **Codespaces** → **Create codespace on main**

---

### Step 2: Set up virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
```

---

### Step 3: Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### Step 4: Run the Flask app
```bash
python app.py
```

Or using Flask CLI:
```bash
export FLASK_APP=app.py
export FLASK_ENV=development
flask run --host=0.0.0.0 --port=5000
```

---

### Step 5: Open the web app
1. Go to the **Ports** tab in Codespaces
2. Open port **5000**
3. Click the forwarded URL to access the app

---

## Run Locally (Windows / macOS / Linux)

### Step 1: Clone the repository
```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_FOLDER>
```

---

### Step 2: Create & activate virtual environment

**macOS / Linux**
```bash
python -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

---

### Step 3: Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### Step 4: Run the app
```bash
python app.py
```

Then open your browser at:
```
http://127.0.0.1:5000
```

(You need to change the URL in the code first.)

---

## Flow of Ingreadyent
1. User uploads an ingredient image
2. Frontend validates file type and size
3. Image is sent to Flask backend via POST request
4. Backend:
   - Converts image to RGB
   - Resizes and normalizes image
   - Runs MobileNetV2 inference
   - Predicts ingredient and freshness
   - Computes confidence score
5. Results are returned as JSON
6. Frontend displays predictions to the user

---

## Model Details
- **Backbone**: MobileNetV2 (pretrained on ImageNet)
- **Architecture**:
  - Shared feature extractor
  - Ingredient classification head (8 classes)
  - Freshness classification head (2 classes)
- **Loss Function**: Combined CrossEntropyLoss
- **Optimizer**: Adam
- **Epochs**: 12
- **Batch Size**: 32

---

## Common Issues

### Model file not found
- Ensure `mobilenetv2_cv_model.pt` is located in the project root
- Run `python app.py` from the root directory

### Port not accessible
- Run Flask with:
```bash
flask run --host=0.0.0.0 --port=5000
```

- Make sure the port is correct in the ```app.py```

### Dependency errors
- Use the exact packages listed in `requirements.txt`
- Upgrade pip before installing dependencies

---

## Limitations
- Dataset size is limited
- Freshness classification is binary (Fresh / Spoiled)
- Model has not been tested with real-world users
- Performance metrics beyond training loss are not included

---

## Future Work
- Expand dataset size and diversity
- Add multi-level freshness categories
- Evaluate model using test-set metrics
- Conduct real-user testing

---

## Authors
- Aurelia Faren Suyanto
- Kenia Esmeralda Ramos Javier
- Stefani Gifta Ganda

## Lecturer
Jaehyeon Park