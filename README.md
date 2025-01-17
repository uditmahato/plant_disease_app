# Plant Disease Detection App

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

Welcome to the **Plant Disease Detection App**! This app uses a deep learning model built with PyTorch to detect diseases in plants based on images of their leaves. Currently, the app supports disease detection for **maize (corn)**, but we plan to expand to more plants in the future.

## 🌱 Live App

You can access the live app here: [diagnoseplantdisease.streamlit.app](https://diagnoseplantdisease.streamlit.app/)

---

## 🚀 Features

- **Disease Detection**: Upload an image of a plant leaf, and the app will predict whether it is healthy or diseased.
- **User-Friendly Interface**: Built with Streamlit for a simple and intuitive user experience.
- **Deep Learning Model**: Powered by a ResNet-based model trained on a dataset of maize leaf images.

---

## 📂 Dataset

The dataset used in this project includes images of plant leaves with and without diseases. The images are preprocessed for efficient training, and data augmentation techniques are applied to improve model generalization.

### Dataset Details:
- **Source**:PlantVillage]([https://plantvillage.psu.edu/](https://github.com/spMohanty/PlantVillage-Dataset/tree/master/raw/color)) (replace with the actual source).
- **Classes**: Healthy, Common Rust, Northern Leaf Blight, etc. (list the specific diseases).
- **Size**: 4188 images (157 MB).

---

## 🧠 Model

The project utilizes **ResNet (Residual Neural Network)**, a state-of-the-art deep learning architecture, for transfer learning. Pretrained weights on ImageNet are leveraged to initialize the model, followed by fine-tuning on the plant disease dataset.

### Key Components:
- **Data Preprocessing and Augmentation**: Images are resized, normalized, and augmented to improve model generalization.
- **Transfer Learning with ResNet**: The ResNet model is fine-tuned on the plant disease dataset for accurate classification.
- **Model Training and Evaluation**: The model is trained using PyTorch and evaluated on a validation set to ensure reliability.

---

## 🎯 Objectives

- Develop a reliable plant disease detection model.
- Use transfer learning to achieve high accuracy with limited data.
- Evaluate model performance and optimize for real-world deployment.

---

## 🛠️ Future Plans

We are actively working on expanding the app to support more plants and diseases. Here’s what’s coming soon:

- **Support for More Plants**: Add disease detection for tomatoes, potatoes, apples, and more.
- **Improved Model**: Train the model on a larger and more diverse dataset for better accuracy.
- **User Feedback**: Allow users to provide feedback on predictions to improve the model over time.

---

## 🧰 Resources

- **Code**: The source code for this app is available on [GitHub](https://github.com/uditmahato/plant_disease_app).
- **Model**: The trained PyTorch model (`model.pth`) is included in the repository.
- **Notebook**: Check out the Jupyter notebook (`corn-leaf-disease-detection-with-resnet-pytorch.ipynb`) for details on how the model was trained.

---

## 🛠️ Installation

To run this app locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/uditmahato/plant_disease_app.git
   cd plant_disease_app

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt

3. Run the app:
   ```bash
   streamlit run maize_plant_disease.py

## 🤝 Contributing
We welcome contributions! If you’d like to contribute to this project, please follow these steps:
- 1.Fork the repository.
- 2.Create a new branch for your feature or bug fix.
- 3.Commit your changes.
- 4.Submit a pull request.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Thanks to [Streamlit](https://streamlit.io/) for the amazing framework.
- Thanks to [PyTorch](https://pytorch.org/) for the deep learning library.
- Special thanks to the creators of the dataset used for training the model.

---

## 📧 Contact

For questions or feedback, feel free to reach out:

- **Udit Mahato**: [GitHub](https://github.com/uditmahato) | [Email](mailto:uditmahato29271@gmail.com)
