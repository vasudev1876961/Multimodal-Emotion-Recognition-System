# 🧠 Multimodal Emotion Recognition System

A complete AI-based system that detects human emotions using **Text + Facial Expressions** in real-time.
This project combines **Natural Language Processing (NLP)** and **Computer Vision (CV)** to improve emotion detection accuracy through a multimodal approach.

---

## 🚀 Features

* 📝 Text Emotion Detection (TF-IDF + Logistic Regression)
* 📷 Real-time Face Emotion Detection (CNN + OpenCV)
* 🔗 Multimodal Fusion (Text + Face)
* 🌐 Web Application (Flask + JavaScript)
* 🎥 Live Webcam inside Browser
* 📊 Clean UI with Emotion Results

---

## 🏗️ Project Structure

```
EmotionAI/
│
├── data/
│   ├── text/
│   └── face/
│
├── src/
│   ├── text/
│   ├── face/
│   ├── fusion/
│
├── models/
│   ├── text_model/
│   └── face_model/
│
├── utils/
│   ├── config.py
│   └── helper.py
│
├── web/
│   ├── app.py
│   ├── templates/
│   └── static/
│
├── main.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Technologies Used

* **Python**
* **NumPy, Pandas**
* **Scikit-learn**
* **TensorFlow / Keras**
* **OpenCV**
* **Flask**
* **HTML, CSS, JavaScript**

---

## 📊 Datasets

* 📌 Text Dataset: Emotion Dataset (Kaggle)
* 📌 Face Dataset: FER-2013

---

## 🧠 Model Details

### 🔹 Text Model

* TF-IDF Vectorization
* Logistic Regression Classifier

### 🔹 Face Model

* CNN (Convolutional Neural Network)
* Input: 48×48 grayscale images
* Output: 7 emotion classes

### 🔹 Fusion Logic

* If both predictions match → return emotion
* If mismatch → prioritize face emotion

---

## 🖥️ How to Run the Project

### 🔹 1. Clone Repository

```
git clone https://github.com/vasudev1876961/EmotionAI.git
cd EmotionAI
```

---

### 🔹 2. Install Requirements

```
pip install -r requirements.txt
```

---

### 🔹 3. Train Models

#### Text Model

```
python src/text/train.py
```

#### Face Model

```
python src/face/train.py
```

---

### 🔹 4. Run Web Application

```
python web/app.py
```

---

### 🔹 5. Open Browser

```
http://127.0.0.1:5000
```

---

## 🎯 Usage

1. Enter text in input field
2. Allow webcam access
3. Click **Analyze Emotion**
4. View:

   * Text Emotion
   * Face Emotion
   * Final Emotion

---

## 📸 Output

* Real-time emotion detection
* Multimodal fusion result
* Clean UI display

---

## 🔥 Advantages

* Improved accuracy using multimodal data
* Real-time interaction
* Scalable architecture
* User-friendly interface

---

## ⚠️ Limitations

* Performance depends on lighting conditions
* Basic NLP model (can be improved with BERT)
* Face detection may fail in complex backgrounds

---

## 🚀 Future Improvements

* Add Voice Emotion Recognition
* Use BERT for text analysis
* Deploy on cloud (AWS / Streamlit Cloud)
* Improve fusion using confidence scores

---

## 👨‍💻 Author

**Vasu Deva**
B.Tech AI & Data Science

---

## ⭐ Acknowledgements

* FER-2013 Dataset
* Kaggle Emotion Dataset
* OpenCV
* TensorFlow

---

## 📌 Conclusion

This project demonstrates how combining multiple modalities (text + face) leads to more accurate and reliable emotion detection systems. It showcases practical applications of AI in human-computer interaction.

---


