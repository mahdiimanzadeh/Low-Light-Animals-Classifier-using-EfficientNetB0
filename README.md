# 🐾 Low-Light Animal Classification with EfficientNetB0

This project focuses on classifying **90 animal species in low-light conditions** using **EfficientNetB0** and **TensorFlow/Keras**. It leverages transfer learning, data augmentation, and best practices like callbacks, caching, and corrupted image filtering.

---

## 📂 Dataset

**Source:** [Kaggle - Low Light Animals Dataset](https://www.kaggle.com/datasets/subratasarkar32/low-light-animals)

#### After downloading, the dataset is extracted to: 
#### 'content/low_light_animals/animals_low_light/animals_low_light'
---

## 🚀 Key Features

- ✅ Transfer learning using **EfficientNetB0** pretrained on ImageNet
- ✅ Supports **90-class classification**
- ✅ **Image enhancement check** (removal of corrupted files)
- ✅ Advanced **data augmentation** with `tf.keras.layers`
- ✅ Training pipeline using `ImageDataGenerator` with augmentation
- ✅ Model checkpointing, early stopping, and learning rate reduction
- ✅ Final visualization of training/validation accuracy & loss

---

## 📦 Dependencies

#### Install with pip:

#### pip install tensorflow opencv-python matplotlib tqdm
---

## ⚙️ Architecture

#### Input (224x224x3)
#### │
#### ├── Data Augmentation (Flip, Rotation, Zoom)
#### ├── EfficientNetB0 (pretrained, frozen)
#### ├── GlobalAveragePooling2D
#### ├── Dense(128) + BatchNorm + Dropout(0.45)
#### ├── Dense(256) + BatchNorm + Dropout(0.45)
#### └── Dense(90, softmax)  ← Output layer
