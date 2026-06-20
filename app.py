from flask import Flask, request
import torch
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from PIL import Image
import os

app = Flask(__name__)

# Class labels (match your dataset folders)
class_labels = ['E-Waste', 'Glass', 'Organic Waste', 'Paper Or Cardboard', 'Plastic']

# Load model
model = models.resnet18(weights=None)  # no pretrained weights here
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, len(class_labels))
model.load_state_dict(torch.load("models/ecowaste_classifier.pth", map_location=torch.device('cpu')))
model.eval()

# Transform for uploaded images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        confidence = torch.softmax(outputs, dim=1)[0][predicted].item() * 100
    return class_labels[predicted], confidence

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filepath = os.path.join("uploads", file.filename)
            os.makedirs("uploads", exist_ok=True)
            file.save(filepath)
            label, confidence = predict_image(filepath)
            return f'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>EcoWaste Classifier Result</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body class="bg-light">
    <div class="container py-5">
        <div class="card shadow-lg p-4 text-center">
            <h2 class="text-success mb-4">♻️ EcoWaste Classifier</h2>
            <div class="alert alert-success fs-5">
                <strong>Prediction:</strong> {label}<br>
                <small class="text-muted">Confidence: {confidence:.2f}%</small>
            </div>
            <a href="/" class="btn btn-outline-success mt-3">🔄 Try Another Image</a>
        </div>
    </div>
</body>
</html>
'''

    return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>EcoWaste Classifier</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body class="bg-light">
    <div class="container py-5">
        <div class="card shadow-lg p-4">
            <h2 class="text-center text-success mb-4">♻️ EcoWaste Classifier </h2>
            <form method="POST" enctype="multipart/form-data" class="text-center">
                <div class="mb-3">
                    <input type="file" name="file" class="form-control">
                </div>
                <button type="submit" class="btn btn-success px-4">Upload & Classify</button>
            </form>
        </div>
    </div>
</body>
</html>
'''


if __name__ == "__main__":
    app.run(debug=True)
