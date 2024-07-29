# Cats vs. Dogs Image Classification

This project aims to classify images of cats and dogs using a convolutional neural network (CNN) built with TensorFlow and Keras. The dataset used is the "Cats vs. Dogs" dataset, which is available through TensorFlow's dataset repository.

## Dataset

The dataset consists of images of cats and dogs, organized into training and validation directories. Each directory contains subdirectories for each class (cats and dogs).

The training Dataset can be found here : https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip


The dataset we have downloaded has the following directory structure:

```
cats_and_dogs_filtered
├── train
│   ├── cats: [cat.0.jpg, cat.1.jpg, cat.2.jpg, ...]
│   ├── dogs: [dog.0.jpg, dog.1.jpg, dog.2.jpg, ...]
├── validation
│   ├── cats: [cat.2000.jpg, cat.2001.jpg, cat.2002.jpg, ...]
│   ├── dogs: [dog.2000.jpg, dog.2001.jpg, dog.2002.jpg, ...]
```

## Data Augumentation

Overfitting often occurs when we have a small number of training examples. One way to fix this problem is to augment our dataset so that it has sufficient number and variety of training examples.
Data augmentation takes the approach of generating more training data from existing training samples, by augmenting the samples through random transformations that yield believable-looking images. 

The goal is that at training time, the model will never see the exact same picture twice. This exposes the model to more aspects of the data, allowing it to generalize better.

![image](https://github.com/user-attachments/assets/e5abb781-20f4-48f3-bb7d-152e17ef6b55)

### Flipping the image horizontally

![image](https://github.com/user-attachments/assets/4110cd9d-590f-4da8-b5c4-870f035ce0c9)

###Rotating the image

![image](https://github.com/user-attachments/assets/c943d23b-bd51-4934-b35c-9a5b2229893e)

###Applying Zoom

![image](https://github.com/user-attachments/assets/0d235ddc-fe54-46e1-98b7-1fc6ad37a498)


## Installation

To get started with this project, you'll need to have TensorFlow and other dependencies installed. You can install the necessary packages using pip:

```bash
pip install tensorflow matplotlib numpy
```

## Project Structure

- **Data Preparation**: Downloads and extracts the dataset, and sets up data generators for training and validation.
- **Model Creation**: Defines and compiles a CNN model for image classification.
- **Training**: Trains the model using the training data and evaluates it on the validation data.
- **Evaluation**: Generates plots for training and validation accuracy and loss, and visualizes predictions on validation images.


## Model Summary

_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 148, 148, 32)      896       
                                                                 
 max_pooling2d (MaxPooling2  (None, 74, 74, 32)        0         
 D)                                                              
                                                                 
 conv2d_1 (Conv2D)           (None, 72, 72, 64)        18496     
                                                                 
 max_pooling2d_1 (MaxPoolin  (None, 36, 36, 64)        0         
 g2D)                                                            
                                                                 
 conv2d_2 (Conv2D)           (None, 34, 34, 128)       73856     
                                                                 
 max_pooling2d_2 (MaxPoolin  (None, 17, 17, 128)       0         
 g2D)                                                            
                                                                 
 conv2d_3 (Conv2D)           (None, 15, 15, 256)       295168    
                                                                 
 max_pooling2d_3 (MaxPoolin  (None, 7, 7, 256)         0         
 g2D)                                                            
                                                                 
 dropout (Dropout)           (None, 7, 7, 256)         0         
                                                                 
 flatten (Flatten)           (None, 12544)             0         
                                                                 
 dense (Dense)               (None, 512)               6423040   
                                                                 
 dense_1 (Dense)             (None, 1)                 513       
                                                                 
=================================================================
Total params: 6811969 (25.99 MB)
Trainable params: 6811969 (25.99 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________


## Results

The trained model achieved an accuracy of **76.90%** on the validation dataset.


##Visualizing results of the training

![image](https://github.com/user-attachments/assets/68f854f0-6e2a-4b9e-b219-c0a49e690dab)


### Example Predictions

Here are 100 images from the validation dataset with their predictions:

![image](https://github.com/user-attachments/assets/29d82fac-2e90-451b-8ae3-2a9c5ae8ea15)


Replace `path/to/predictions_image.png` with the path to the image showing predictions.

## Usage

1. **Prepare the Dataset**: Ensure the dataset is downloaded and extracted properly.
2. **Run the Code**: Execute the script to train the model and evaluate its performance.
3. **View Results**: Check the generated plots and predictions to analyze the model's performance.

## Saving and Downloading the Model

The trained model is saved as `CatsVsDogs_Model.keras` and can be downloaded from:

[Download CatsVsDogs_Model.keras](https://drive.google.com/drive/folders/1l1cikFWNjV_LdurGFtYmf199IiW6JkBE?usp=sharing)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

