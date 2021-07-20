import geffnet
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image  
import torch.nn as nn
from flask import Flask, request, render_template

transform = transforms.Compose([transforms.Resize(size=(224,224)),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])

def get_prediction(image):
    number_to_class = {0: 'Biryani',1: 'Butter Naan',2: 'Chai',3: 'Chole Bhature',4: 'Dhokla',5: 'Gulab Jamun',6: 'Jalebi',7: 'Momos',8: 'Paneer Sabzi',9: 'Pav Bhaji',10: 'Rasgulla',11: 'Samosa'}
    with torch.no_grad():
        model.eval()
        # Model outputs log probabilities
        out = model(image)
        ps = F.softmax(out, dim =1)
        return number_to_class[int(torch.argmax(ps).item())]


app = Flask(__name__)

model = geffnet.create_model('efficientnet_b2', pretrained=True)
model.classifier = nn.Sequential(nn.Linear(1408,512),
                           nn.ReLU(),
                           nn.Dropout(p=0.4),
                           nn.Linear(512,128),
                           nn.ReLU(),
                           nn.Dropout(p=0.4),
                           nn.Linear(128,12))
model.load_state_dict(torch.load("model.pt", map_location=torch.device('cpu')))
model.eval()
print('Model Loaded!')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        file.save('img.jpg')
        image = Image.open('img.jpg').convert('RGB')
        transformed_image = transform(image).unsqueeze(0)
        prediction = get_prediction(transformed_image)
    return prediction


if __name__ == '__main__':
    app.run(debug = True)