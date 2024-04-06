import os
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import requests
from flask import Flask, render_template, request, send_from_directory
from PIL import Image
import torch
from torchvision import transforms
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext101_32x4d_ssl')

# Define the path to the saved model
model_path = "files_req/alltrained_model.pth"

# Load the pre-trained model
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

df = pd.read_csv('./files_req/leafsnap-dataset-images.txt', delimiter='\t')
# df = df.loc[:10733, :]
classes_labels = df['species'].unique()
leaf = " leaf"

# Define the transform to preprocess the input image
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define a function to perform inference
def predict_image(image_path):
    image = Image.open(image_path)
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    with torch.no_grad():
        output = model(input_batch)
    return output

# Define a function to get random images from Unsplash API
# def get_random_images(query, count=6):
#     unsplash_access_key = "0AsIWfJKr2jerWdR78OYbWKAHFgBQFYedwC4hV3IJvE"
#     url = f"https://api.unsplash.com/photos/random?query={query}&count={count}"
#     headers = {
#         "Accept-Version": "v1",
#         "Authorization": f"Client-ID {unsplash_access_key}"
#     }
#     response = requests.get(url, headers=headers)
#     if response.status_code == 200:
#         data = response.json()
#         return [item["urls"]["regular"] for item in data]
#     else:
#         return []

def get_top_images(query, count=6):
    unsplash_access_key = "0AsIWfJKr2jerWdR78OYbWKAHFgBQFYedwC4hV3IJvE"
    url = f"https://api.unsplash.com/search/photos?query={query}&per_page={count}&client_id={unsplash_access_key}"
    headers = {
        "Accept-Version": "v1"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        return [photo["urls"]["regular"] for photo in data["results"]]
    else:
        return []


@app.route('/home')
def index():
    return render_template('index.html') 
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Define a route to handle the home page
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return render_template('index.html', message='No file uploaded')
        
        file = request.files['file']
        
        # Check if the file is empty
        if file.filename == '':
            return render_template('index.html', message='No file selected')
        
        # Check if the file is valid
        if file:
            # Save the uploaded file
            filename = secure_filename(file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)
            image_path = image_path.replace("\\", "/")
            
            try:
                # Perform prediction
                output = predict_image(image_path)
                probabilities = torch.softmax(output, dim=1)
                # Process the output and get the predicted class
                predicted_index = torch.argmax(probabilities, dim=1)
                predicted_class = classes_labels[predicted_index[0]] 
                query = (predicted_class + leaf).split()
                query = "+".join(query)
                # Get 2 to 5 random images related to the predicted class
                random_images = get_top_images(query=query, count=6)
                
                return render_template('index.html', uploaded_image=image_path, message='Prediction: ' + predicted_class, images=random_images)
            
            except Exception as e:
                return render_template('index.html', uploaded_image=image_path,message="Upload the leaf image please")
    
    return render_template('index.html')




if __name__ == '__main__':
    app.run()
