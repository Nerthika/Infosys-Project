import os
import tensorflow as tf
import numpy as np
from flask import Flask, request, render_template, redirect, jsonify
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Set up the upload folder and allowed file extensions
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Disease labels for classification
DISEASE_LABELS = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
    'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
    'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites_Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]
 #Sample disease information (expand as per your actual model and needs)
disease_info = {
   "Apple___Apple_scab": {
       "name": "Apple Scab",
       "cause": "Fungal infection caused by the pathogen *Venturia inaequalis*.",
       "prevention": "Remove infected leaves and debris. Use fungicides in early spring.",
       "suggested_products": "Fungicides containing myclobutanil, mancozeb."
   },
   "Apple___Black_rot": {
       "name": "Apple Black Rot",
       "cause": "Fungal infection caused by *Botryosphaeria obtusa*.",
       "prevention": "Prune infected branches and clean up fallen fruit. Use fungicides.",
       "suggested_products": "Mancozeb, chlorothalonil."
   },
   "Apple___Cedar_apple_rust": {
       "name": "Apple Cedar Apple Rust",
       "cause": "Caused by the fungus *Gymnosporangium juniperi-virginianae*.",
       "prevention": "Remove infected leaves and trees. Apply fungicides during spring.",
       "suggested_products": "Neem oil, copper-based fungicides."
   },
   "Apple___healthy": {
       "name": "Healthy Apple",
       "cause": "No disease detected.",
       "prevention": "Ensure proper care with regular pruning and watering.",
       "suggested_products": "N/A"
   },
   "Blueberry___healthy": {
       "name": "Healthy Blueberry",
       "cause": "No disease detected.",
       "prevention": "Ensure proper watering and soil pH management.",
       "suggested_products": "N/A"
   },
   "Cherry_(including_sour)___Powdery_mildew": {
       "name": "Cherry Powdery Mildew",
       "cause": "Fungal infection caused by *Podosphaera clandestina*.",
       "prevention": "Prune infected areas, remove fallen leaves, and apply fungicides.",
       "suggested_products": "Sulfur-based fungicides."
   },
   "Cherry_(including_sour)___healthy": {
       "name": "Healthy Cherry",
       "cause": "No disease detected.",
       "prevention": "Proper care and pruning will keep the plant healthy.",
       "suggested_products": "N/A"
   },
   "Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot": {
       "name": "Cercospora Leaf Spot & Gray Leaf Spot",
       "cause": "Fungal infection caused by *Cercospora* species.",
       "prevention": "Apply fungicides and ensure proper spacing for air circulation.",
       "suggested_products": "Azoxystrobin, copper-based fungicides."
   },
   "Corn_(maize)___Common_rust_": {
       "name": "Common Rust",
       "cause": "Fungal infection caused by *Puccinia sorghi*.",
       "prevention": "Remove infected plants and use rust-resistant varieties.",
       "suggested_products": "Azoxystrobin, tebuconazole."
   },
   "Corn_(maize)___Northern_Leaf_Blight": {
       "name": "Northern Leaf Blight",
       "cause": "Fungal infection caused by *Exserohilum turcicum*.",
       "prevention": "Use resistant corn varieties, rotate crops, and apply fungicides.",
       "suggested_products": "Mancozeb, chlorothalonil."
   },
   "Corn_(maize)___healthy": {
       "name": "Healthy Corn",
       "cause": "No disease detected.",
       "prevention": "Maintain healthy soil and water management.",
       "suggested_products": "N/A"
   },
   "Grape___Black_rot": {
       "name": "Grape Black Rot",
       "cause": "Fungal infection caused by *Guignardia bidwellii*.",
       "prevention": "Remove infected fruit and leaves. Apply fungicides during growth.",
       "suggested_products": "Fungicides containing thiophanate-methyl."
   },
   "Grape___Esca_(Black_Measles)": {
       "name": "Grape Esca (Black Measles)",
       "cause": "Fungal infection caused by *Phaeoacremonium* species.",
       "prevention": "Prune infected areas, remove diseased wood, and apply fungicides.",
       "suggested_products": "Fungicides with azoxystrobin, fludioxonil."
   },
   "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
       "name": "Grape Leaf Blight (Isariopsis Leaf Spot)",
       "cause": "Fungal infection caused by *Isariopsis* species.",
       "prevention": "Remove infected leaves, and apply fungicides.",
       "suggested_products": "Mancozeb, copper-based fungicides."
   },
   "Grape___healthy": {
       "name": "Healthy Grape",
       "cause": "No disease detected.",
       "prevention": "Ensure proper care with pruning and spacing.",
       "suggested_products": "N/A"
   },
   "Orange___Haunglongbing_(Citrus_greening)": {
       "name": "Citrus Greening (Huanglongbing)",
       "cause": "Bacterial infection caused by *Candidatus Liberibacter asiaticus*.",
       "prevention": "Remove infected trees, use resistant varieties.",
       "suggested_products": "N/A (no cure, prevention through tree removal)"
   },
   "Peach___Bacterial_spot": {
       "name": "Peach Bacterial Spot",
       "cause": "Bacterial infection caused by *Xanthomonas campestris*.",
       "prevention": "Prune infected areas, apply copper-based fungicides.",
       "suggested_products": "Copper sulfate, copper oxychloride."
   },
   "Peach___healthy": {
       "name": "Healthy Peach",
       "cause": "No disease detected.",
       "prevention": "Proper pruning, pest management, and irrigation.",
       "suggested_products": "N/A"
   },
   "Pepper,_bell___Bacterial_spot": {
       "name": "Bell Pepper Bacterial Spot",
       "cause": "Bacterial infection caused by *Xanthomonas campestris*.",
       "prevention": "Use resistant varieties, apply copper-based products.",
       "suggested_products": "Copper sulfate, chlorothalonil."
   },
   "Pepper,_bell___healthy": {
       "name": "Healthy Bell Pepper",
       "cause": "No disease detected.",
       "prevention": "Water regularly, prune, and protect from pests.",
       "suggested_products": "N/A"
   },
   "Potato___Early_blight": {
       "name": "Early Blight",
       "cause": "Fungal infection caused by *Alternaria solani*.",
       "prevention": "Rotate crops, use fungicides.",
       "suggested_products": "Chlorothalonil, mancozeb."
   },
   "Potato___Late_blight": {
       "name": "Late Blight",
       "cause": "Fungal infection caused by *Phytophthora infestans*.",
       "prevention": "Apply fungicides, destroy infected plants.",
       "suggested_products": "Mancozeb, copper fungicides."
   },
   "Potato___healthy": {
       "name": "Healthy Potato",
       "cause": "No disease detected.",
       "prevention": "Regular watering, pest control, and crop rotation.",
       "suggested_products": "N/A"
   },
   "Raspberry___healthy": {
       "name": "Healthy Raspberry",
       "cause": "No disease detected.",
       "prevention": "Proper spacing, pruning, and pest management.",
       "suggested_products": "N/A"
   },
   "Soybean___healthy": {
       "name": "Healthy Soybean",
       "cause": "No disease detected.",
       "prevention": "Maintain soil health, use resistant varieties.",
       "suggested_products": "N/A"
   },
   "Squash___Powdery_mildew": {
       "name": "Squash Powdery Mildew",
       "cause": "Fungal infection caused by *Erysiphe cichoracearum*.",
       "prevention": "Remove infected leaves, apply sulfur-based fungicides.",
       "suggested_products": "Sulfur, neem oil."
   },
   "Strawberry___Leaf_scorch": {
       "name": "Strawberry Leaf Scorch",
       "cause": "Bacterial infection caused by *Xanthomonas fragariae*.",
       "prevention": "Prune infected leaves, apply copper-based fungicides.",
       "suggested_products": "Copper sulfate, copper oxychloride."
   },
   "Strawberry___healthy": {
       "name": "Healthy Strawberry",
       "cause": "No disease detected.",
       "prevention": "Ensure proper watering, pest management.",
       "suggested_products": "N/A"
   },
   "Tomato___Bacterial_spot": {
       "name": "Tomato Bacterial Spot",
       "cause": "Bacterial infection caused by *Xanthomonas campestris*.",
       "prevention": "Prune affected areas, apply copper-based fungicides.",
       "suggested_products": "Copper sulfate, chlorothalonil."
   },
   "Tomato___Early_blight": {
       "name": "Tomato Early Blight",
       "cause": "Fungal infection caused by *Alternaria solani*.",
       "prevention": "Rotate crops, apply fungicides.",
       "suggested_products": "Chlorothalonil, mancozeb."
   },
   "Tomato___Late_blight": {
       "name": "Tomato Late Blight",
       "cause": "Fungal infection caused by *Phytophthora infestans*.",
       "prevention": "Apply fungicides, destroy infected plants.",
       "suggested_products": "Mancozeb, copper fungicides."
   },
   "Tomato___Leaf_Mold": {
       "name": "Tomato Leaf Mold",
       "cause": "Fungal infection caused by *Cladosporium fulvum*.",
       "prevention": "Prune infected areas, improve ventilation.",
       "suggested_products": "Azoxystrobin, tebuconazole."
   },
   "Tomato___Septoria_leaf_spot": {
       "name": "Tomato Septoria Leaf Spot",
       "cause": "Fungal infection caused by *Septoria lycopersici*.",
       "prevention": "Prune infected leaves, use fungicides.",
       "suggested_products": "Chlorothalonil, mancozeb."
   },
   "Tomato___healthy": {
       "name": "Healthy Tomato",
       "cause": "No disease detected.",
       "prevention": "Ensure healthy soil, proper spacing, and pest management.",
       "suggested_products": "N/A"
   }
}


# Load the trained model from Google Drive
model_path = 'Plantdiseases_model.h5'  # Replace with your model path
model = tf.keras.models.load_model(model_path)

# Function to check if the uploaded file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to preprocess the uploaded image
def preprocess_image(image_path, target_size=(224, 224)):
    img = image.load_img(image_path, target_size=target_size)  # Resize image to match model's input size
    img_array = image.img_to_array(img)  # Convert the image to an array
    img_array = img_array / 255.0  # Normalize the image (scaling pixel values to [0, 1])
    img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension (model expects a batch)
    return img_array

def get_disease_info(disease_name):
    return disease_info.get(disease_name, "No information available.")

# Function to make a prediction
def predict_disease(image_path):
    # Preprocess the image
    img_array = preprocess_image(image_path)

    # Make a prediction
    prediction = model.predict(img_array)

    # Get the index of the highest probability
    predicted_class_index = np.argmax(prediction, axis=1)[0]

    # Map the index to the corresponding disease name
    predicted_disease = DISEASE_LABELS[predicted_class_index]

    # Output the predicted disease to the console (or prompt)
    print(f"Predicted Disease: {predicted_disease}")  # This will print the output in the server console

    # Get disease info
    disease_info = get_disease_info(predicted_disease)

    # Return the predicted disease and its information
    return predicted_disease, disease_info

@app.route('/', methods=['GET', 'POST'])
def home():
    predicted_disease = None
    disease_info = None

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            # Save the uploaded image
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Make a prediction using the trained model
            predicted_disease, disease_info = predict_disease(file_path)

            # Log for debugging
            print(f"Predicted Disease: {predicted_disease}")
            print(f"Disease Info: {disease_info}")

            # Return the prediction and disease info to be handled in the frontend
            return jsonify({
                'predicted_disease': predicted_disease,
                'disease_info': disease_info
            })

    # Return the page with the prediction and disease info
    return render_template('index.html', predicted_disease=predicted_disease, disease_info=disease_info)

@app.route('/about')
def about():
    return render_template('aboutus.html')

@app.route('/agrisupport')
def agrisupport():
    return render_template('agrisupport.html')

@app.route('/blogs')
def blogs():
    return render_template('blog.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/feedback')
def feedback():
    return render_template('feedback.html')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
