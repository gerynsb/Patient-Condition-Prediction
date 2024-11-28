from flask import Flask, render_template, request
from googletrans import Translator
import pickle

app = Flask(__name__)

# Inisialisasi translator
translator = Translator()

# Muat model dan vectorizer dari file
with open('model/model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('model/vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Ambil input gejala dari form
        input_text_id = request.form['symptom']

        # Terjemahkan ke bahasa Inggris
        input_text_en = translator.translate(input_text_id, src='id', dest='en').text

        # Transformasi teks dan prediksi
        test = vectorizer.transform([input_text_en])
        prediction = model.predict(test)[0]

        return render_template('index.html', symptom=input_text_id, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
