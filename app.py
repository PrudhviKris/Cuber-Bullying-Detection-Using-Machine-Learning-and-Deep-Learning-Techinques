from flask import Flask, render_template, request
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load stopwords
with open (r"C:\Users\prudh\Downloads\Project\CyberBullying_Detection\stopwords.txt", "r") as file:
    stopwords = file.read().splitlines()

# Load vectorizer and model
vectorizer = TfidfVectorizer(stop_words=stopwords, lowercase=True, vocabulary=pickle.load(open(r"C:\Users\prudh\Downloads\Project\CyberBullying_Detection\tfidfvectoizer.pkl", "rb")))
model = pickle.load(open(r"C:\Users\prudh\Downloads\Project\CyberBullying_Detection\LinearSVCTuned.pkl", 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/verify', methods=['GET', 'POST'])
def verify():
    prediction = None
    if request.method == 'POST':
        user_input = request.form['text']
        transformed_input = vectorizer.fit_transform([user_input])
        pred = model.predict(transformed_input)[0]
        
        if pred == 1:
            prediction = "🚨 This text contains Bullying Words"
        else:
            prediction = "✅ This text is Non-Bullying"
    
    return render_template('verify.html', prediction=prediction)

@app.route('/helpdesk', methods=['GET', 'POST'])
def helpdesk():
    if request.method == 'POST':
        email = request.form['email']
        query = request.form['query']
        # Here you can later add logic to save or send email
        return render_template('helpdesk.html', message="Thank you for contacting us! We'll get back soon.")
    
    return render_template('helpdesk.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
