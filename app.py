from flask import Flask, render_template, request
from torchvision import models, transforms
from PIL import Image
import torch
import torch.nn as nn
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('resnet18_skin_cancer.pth', map_location=torch.device('cpu')))
model.eval()

# Define image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                              'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    image_path = None

    if request.method == 'POST':
        file = request.files.get('image')  # Safe access
        if file and file.filename:
            image_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(image_path)

            image = Image.open(image_path).convert('RGB')
            img_tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                output = model(img_tensor)
                _, predicted = torch.max(output, 1)

            if predicted.item() == 0:
                result = {
                    "label": "✅ You are Safe – No Cancer Detected",
                    "type": "benign",
                    "emoji": "😊",
                    "color": "green"
                }
            else:
                result = {
                    "label": "❗ Warning: Possible Skin Cancer Detected!",
                    "type": "malignant",
                    "emoji": "⚠️",
                    "color": "red"
                }

    return render_template('index.html', result=result, image_path=image_path)

if __name__ == '__main__':
    app.run(debug=False)
