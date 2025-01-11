# Forest_Fire_Detection
"A forest fire detection project using TensorFlow to train a model for identifying fire occurrences from imagery, and deploying the solution with Streamlit for real-time monitoring and prediction."

# 🌳 Forest Fire Detection
<img src="https://github.com/rpjinu/Forest_Fire_Detection/blob/main/Forest_fire_image.jpg" width="800">

## 📚 Project Description
This project aims to build an end-to-end solution for detecting 🌳⚡️ fires using deep learning. The solution involves training a 💀 Convolutional Neural Network (CNN) using TensorFlow to classify 🌐 images as either containing a 🌳⚡️ fire or not. Once the model is trained, it is deployed using 🌐 Streamlit, enabling real-time 🕓 monitoring and 🔀 predictions from new 🌐 image data.

## 📁 Project Structure
```
Forest-Fire-Detection/
💀 data/
🔵   🔵 train/                # 🎒 Training dataset
🔵   🔵   🔵 fire/             # 🔥 Images of forest fires
🔵   🔵   🔵 no_fire/          # 🌳 Images without forest fires
🔵   🔵 test/                 # 🌐 Testing dataset
🔵       🔵 fire/             # 🔥 Test images of forest fires
🔵       🔵 no_fire/          # 🌳 Test images without forest fires
🔵
📁 notebooks/
🔵   🔵 EDA.ipynb             # 🌐 Exploratory Data Analysis
📁 models/
🔵   🔵 forest_fire_model.h5  # 💀 Trained model file
📁 scripts/
🔵   🔵 train_model.py        # 🔧 Script for training the CNN model
🔵   🔵 evaluate_model.py     # 🔧 Script for evaluating the model
📁 app/
🔵   🔵 app.py                # 🌐 Streamlit app for deployment
📁 requirements.txt          # 📓 List of dependencies
📁 README.md                 # 📄 Project overview and instructions
📁 LICENSE                   # 🔒 License information
```

## 🔎 Key Steps
### Dataset Link
- [Forest Fire Dataset]([https://www.kaggle.com/datasets/rpjinu/forest-fire-dataset])

### 1. 📊 Data Collection & Preprocessing
- 🔄 Gather and organize the dataset into `fire` and `no_fire` categories.
- 📝 Perform 🌐 image preprocessing such as resizing and normalization.

### 2. 🌐 Exploratory Data Analysis (EDA)
- 🔀 Analyze the dataset to understand class distributions, visualize sample 🌐 images, and identify potential issues.

### 3. 🔧 Model Development
- 💻 Build a CNN using TensorFlow/Keras for 🌐 image classification.
- 💪 Train the model on the 🎒 training dataset, tuning hyperparameters as needed.
- 🔢 Evaluate the model's performance on the 🌐 test dataset.

### 4. 🌐 Model Deployment
- 🌐 Develop a Streamlit app that allows users to upload 🌐 images and get real-time 🕓 predictions on 🌳⚡️ fire presence.
- 🌐 Deploy the Streamlit app to a 🛏️ cloud platform for accessibility.

### 5. 📝 Documentation
- 📝 Include detailed documentation in `README.md` covering the project's purpose, installation instructions, usage, and results.

### 6. 🛠️ Version Control
- 🔧 Use Git for version control, ensuring code and models are tracked and managed efficiently.

## 🔎 Usage Instructions
### 1. 📑 Clone the Repository
```bash
git clone https://github.com/username/Forest-Fire-Detection.git
cd Forest-Fire-Detection
```

### 2. 🔄 Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. 💪 Train the Model
```bash
python scripts/train_model.py
```

### 4. 🌐 Run the Streamlit App
```bash
streamlit run app/app.py
```

## 🔧 Technologies Used
- **TensorFlow**: For building and training the CNN model.
- **Streamlit**: For deploying the interactive 🌐 web app.
- **Python**: Core programming language for scripts and app development.
- **Git**: Version control system for managing project files.

## 🌐 Conclusion
This project demonstrates a complete machine learning pipeline from 🌐 data preprocessing, 💪 model training, and 🔢 evaluation to 🌐 deployment, providing a robust solution for 🌳⚡️ fire detection using deep learning techniques.


