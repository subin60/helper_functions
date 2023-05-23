import numpy as np
import random
import pandas as pd
import os
import zipfile
import matplotlib.pyplot as plt
import datetime
import matplotlib.image as mpimg

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import clone_model
from tensorflow.keras.callbacks import ModelCheckpoint


import os
import subprocess

def install_tensorflow_version(version="2.9.0"):
    """
    This function installs a specific version of TensorFlow using pip and 
    suppresses all the output messages except for the installed TensorFlow version.

    Parameters:
    version (str): The version number of TensorFlow to install. Defaults to "2.9.0".
    
    Returns:
    None
    """
    # Construct the pip install command
    pip_command = f"pip install -U -q tensorflow=={version}"
    
    # Run the pip install command and suppress output
    FNULL = open(os.devnull, 'w')
    subprocess.call(pip_command, stdout=FNULL, stderr=subprocess.STDOUT, shell=True)
    
    # After installation, import tensorflow and print the installed version
    import tensorflow as tf
    print(f'TensorFlow version: {tf.__version__}')



import matplotlib.pyplot as plt

def plot_history(history):
  """
  Plots the training and validation loss and accuracy of a trained model.

  Args:
  history: History object. Outputs of the fit() function of a keras model.

  Returns:
  Two plots. One for loss and one for accuracy.
  """
  
  # Importing the necessary library for plotting
  
  
  # Extracting loss and accuracy for both training and validation from the History object
  loss = history.history['loss']
  accuracy = history.history['accuracy']
  val_loss = history.history['val_loss']
  val_accuracy = history.history['val_accuracy']

  # Getting the number of epochs the model was trained for
  epochs = range(len(history.history['loss']))

  # Plotting training and validation loss
  plt.figure()
  plt.plot(epochs, loss, label='training loss')
  plt.plot(epochs, val_loss, label='validation loss')
  plt.title('Loss')  # Title of the plot
  plt.xlabel('Epochs')  # X-axis label
  plt.ylabel('Loss')  # Y-axis label
  plt.legend()  # Legend to differentiate between training and validation loss

  # Plotting training and validation accuracy
  plt.figure()
  plt.plot(epochs, accuracy, label='training accuracy')
  plt.plot(epochs, val_accuracy, label='validation accuracy')
  plt.title('Accuracy')  # Title of the plot
  plt.xlabel('Epochs')  # X-axis label
  plt.ylabel('Accuracy')  # Y-axis label
  plt.legend()  # Legend to differentiate between training and validation accuracy


import os
import zipfile
  
def download_and_unzip(filepath):
  """
  Downloads and unzips a zip file from a specified filepath.

  Args:
  filepath: A string specifying the URL of the zip file to download.

  Returns:
  None.
  """
  # Import necessary libraries
 

  # Use wget to download the zip file
  os.system(f'wget {filepath}')
  
  # Use os.path.basename to get the filename (with extension) from the filepath
  filename_with_extension = os.path.basename(filepath)
  
  # Create a ZipFile object
  zip_ref = zipfile.ZipFile(filename_with_extension, 'r')

  # Extract all the contents of the zip file in current directory
  zip_ref.extractall()

  # Close the ZipFile object
  zip_ref.close()


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import random

def view_random_image(target_dir, target_class):
  """
  Picks and displays a random image from a specified directory and class.

  Parameters:
  target_dir : str
      The target directory where the image classes directories are.
  target_class : str
      The target class from which to pick the image.

  Returns:
  img : numpy.ndarray
      The image array in RGB format.
  """

  # Setup the target directory (we'll view images from here)
  target_folder = os.path.join(target_dir, target_class)

  # Get a random image path from the target directory
  random_image = random.choice(os.listdir(target_folder))

  # Read in the image using matplotlib
  img = mpimg.imread(os.path.join(target_folder, random_image))

  # Plot the image
  plt.imshow(img)
  plt.title(target_class)
  plt.axis("off")

  print(f"Image shape: {img.shape}") # show the shape of the image

  return img

'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def pred_and_plot(model, filename, class_names, img_shape=224):
  """
  Function to predict the class of an image and plot it. 
  Args:
  model: TensorFlow model
  filename: string, path to the target image
  class_names: list, contains the class names that the model can predict
  img_shape: int, the size of the image the model was trained on (default is 224)

  Returns:
  None, but prints out the predicted class and an image plot
  """
  

  # Read in the image file
  img = tf.io.read_file(filename)

  # Decode the read file into a tensor and resize it to the img_shape
  img = tf.image.decode_image(img)
  img = tf.image.resize(img, size=[img_shape, img_shape])

  # Rescale the image (divide by 255)
  img = img/255.

  # Expand the dimensions of the image tensor from [img_shape, img_shape, color_channels] 
  # to [1, img_shape, img_shape, color_channels] as the model expects a batch
  img_expanded = tf.expand_dims(img, axis=0) 

  # Make a prediction on the image using the model
  pred = model.predict(img_expanded)

  # Get the predicted class
  if len(pred[0]) > 1: # multi-class
    pred_class = class_names[np.argmax(pred)]
  else: # binary class
    pred_class = class_names[int(tf.round(pred))]

  # Plot the image with predicted class as title
  plt.imshow(img)
  plt.title(f"Prediction: {pred_class}")
  plt.axis(False)
  plt.show()
'''
  
import pathlib
import numpy as np

def get_class_names(train_dir):
  """
  Function to get class names from a directory.

  Args:
  train_dir: str, path to the training directory containing class subdirectories.

  Returns:
  class_names: numpy array, array of class names sorted in alphabetical order.
  """

  # Import necessary libraries

  # Convert input to a pathlib Path object (this allows for handy methods to be used on the input)
  data_dir = pathlib.Path(train_dir)

  # Use the glob method to find all class subdirectories, get their names and sort them
  class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))

  # Return class names
  return class_names



# Import necessary libraries
import tensorflow as tf
import datetime

def create_tensorboard_callback(dir_name, experiment_name):
  """
  Creates a TensorBoard callback instance to store log files.

  This function generates a TensorBoard callback which is designed to be used 
  with a TensorFlow Keras model. The callback will write logs for TensorBoard 
  which allow you to visualize dynamic graphs of your training and test 
  metrics, as well as activation histograms for the different layers in your model.

  Args:
    dir_name (str): Target directory to store TensorBoard log files.
    experiment_name (str): Name of the experiment directory (e.g., 'efficientnet_model_1').

  Returns:
    tensorboard_callback (tf.keras.callbacks.TensorBoard): A TensorBoard callback instance 
    configured with the log directory.
  
  Example:
    tensorboard_cb = create_tensorboard_callback("tensorboard_logs", "exp1")
    model.fit(X_train, y_train, callbacks=[tensorboard_cb])
  """
  
  # Combine directory name, experiment name, and current time to form a log directory
  log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  
  # Create a TensorBoard callback instance
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
  
  print(f"Saving TensorBoard log files to: {log_dir}")
  
  # Return the TensorBoard callback instance
  return tensorboard_callback

import os

def walk_through_dir(dir_path):
    """
    Walks through a directory, printing out the number of directories and 
    images (files) in each, along with each subdirectory's name.
    
    Args:
        dir_path (str): The path of the target directory to walk through.
    
    Returns:
        None. However, as a side effect, this function will print:
        - The number of subdirectories in `dir_path`
        - The number of images (or files) in each subdirectory
        - The name of each subdirectory
    """
    # Use os.walk to generate a 3-tuple for each directory it traverses 
    for dirpath, dirnames, filenames in os.walk(dir_path):
        # Get the count of directories and files
        num_dirs = len(dirnames)
        num_files = len(filenames)
        
        # Print the counts and the current directory path
        print(f"There are {num_dirs} directories and {num_files} images in '{dirpath}'.")

        
import matplotlib.pyplot as plt

def compare_history(original_history, new_history, initial_epochs=5):
    """
    This function compares the training history of two different training runs, 
    and plots their accuracy and loss for training and validation sets.
    Useful for visualizing the effects of fine-tuning the model.

    Parameters:
    original_history (History): History object of the original model. 
                                Contains the training and validation loss and accuracy for each epoch.
    new_history (History): History object of the new model (after fine-tuning). 
                           Contains the training and validation loss and accuracy for each epoch.
    initial_epochs (int): Number of epochs the original model was trained for, 
                          helps to align the two plots correctly.

    Returns:
    None
    """
    # Get original history measurements
    acc = original_history.history["accuracy"]
    loss = original_history.history["loss"]

    val_acc = original_history.history["val_accuracy"]
    val_loss = original_history.history["val_loss"]

    # Combine original history with new history
    total_acc = acc + new_history.history["accuracy"]
    total_loss = loss + new_history.history["loss"]

    total_val_acc = val_acc + new_history.history["val_accuracy"]
    total_val_loss = val_loss + new_history.history["val_loss"]

    # Make plots
    plt.figure(figsize=(8, 8))
    
    # Plot training and validation accuracy
    plt.subplot(2, 1, 1)
    plt.plot(total_acc, label='Training Accuracy')
    plt.plot(total_val_acc, label='Validation Accuracy')
    plt.plot([initial_epochs-1, initial_epochs-1], plt.ylim(), label='Start Fine Tuning') 
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    
    # Plot training and validation loss
    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label='Training Loss')
    plt.plot(total_val_loss, label='Validation Loss')
    plt.plot([initial_epochs-1, initial_epochs-1], plt.ylim(), label='Start Fine Tuning') 
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    
    # Show the plot
    plt.show()        

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import random
import tensorflow as tf

def view_augmented_image(directory, class_names, data_augmentation):
    """
    Function to view an original and augmented image from a randomly selected class.

    Parameters:
    directory (str): The path to the directory containing the class subdirectories.
    class_names (list): A list of class names, corresponding to subdirectory names in the directory path.
    data_augmentation (callable): A function or model that performs image augmentation.

    Returns:
    None
    """
    # Choose a random class
    target_class = random.choice(class_names)
    
    # Create the target directory path
    target_dir = os.path.join(directory, target_class)
    
    # Choose a random image from target directory
    random_image = random.choice(os.listdir(target_dir))
    
    # Create the chosen random image path
    random_image_path = os.path.join(target_dir, random_image)
    
    # Read in the chosen target image
    img = mpimg.imread(random_image_path)
    
    # Plot the target image
    plt.imshow(img)
    plt.title(f"Original random image from class: {target_class}")
    plt.axis('off')
    plt.show()

    # Perform data augmentation 
    # The data augmentation model requires input shape (None, height, width, 3)
    augmented_img = data_augmentation(tf.expand_dims(img, axis=0))
    
    # Plot the augmented image
    # The augmented image requires normalization after augmentation
    plt.imshow(tf.squeeze(augmented_img)/255.)
    plt.title(f"Augmented random image from class: {target_class}")
    plt.axis('off')
    plt.show()

    
def print_layer_status(model):
    """
    Function to print the status of the layers in a model.

    Parameters:
    model (tensorflow.python.keras.engine.functional.Functional): The model whose layers' status you want to print.

    Returns:
    None
    """
    trainable_count = 0
    non_trainable_count = 0

    for i, layer in enumerate(model.layers):
        print(f"Layer {i} | Name: {layer.name} | Trainable: {layer.trainable}")
        
        if layer.trainable:
            trainable_count += 1
        else:
            non_trainable_count += 1

    print(f"\nNumber of trainable layers: {trainable_count}")
    print(f"Number of non-trainable layers: {non_trainable_count}")


import tensorflow as tf    

def create_checkpoint_callback(directory_path, monitor="val_accuracy"):
    """
    Creates a ModelCheckpoint callback that saves the best model during training.
    
    Args:
        directory_path (str): Directory where the model checkpoints will be saved.
        monitor (str): Quantity to monitor for choosing the best model to save.
            Default is "val_accuracy".
            
    Returns:
        A tf.keras.callbacks.ModelCheckpoint instance.
    """
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=directory_path,
        save_weights_only=True, # Save only the model weights
        monitor=monitor, # Save the model weights which score the best on the chosen quantity
        save_best_only=True, # Only keep the model that has achieved the "best" so far in terms of monitored quantity
        verbose=1, # Print the details about the saved checkpoints
    )
    
    return checkpoint_callback
  
from tensorflow.keras import layers, Sequential

def create_data_augmentation(rotation_range=0.2, zoom_range=0.2, height_range=0.2, 
                             width_range=0.2, flip_mode="horizontal", include_rescaling=False, name="data_augmentation"):
    """
    Creates a Sequential model for data augmentation.
    
    Args:
        rotation_range (float): A positive float represented as fraction of 2pi, the total range to randomly rotate images.
            Default is 0.2.
        zoom_range (float): A positive float represented as fraction, the total range to randomly zoom images.
            Default is 0.2.
        height_range (float): A positive float represented as fraction, the total range to randomly alter the height of images.
            Default is 0.2.
        width_range (float): A positive float represented as fraction, the total range to randomly alter the width of images.
            Default is 0.2.
        flip_mode (str): One of {"horizontal", "vertical", "horizontal_and_vertical"}.
            "horizontal": Randomly flip inputs horizontally.
            "vertical": Randomly flip inputs vertically.
            "horizontal_and_vertical": Randomly flip inputs both horizontally and vertically.
            Default is "horizontal".
        include_rescaling (bool): Whether to include a Rescaling layer. 
            This layer is needed for some models like ResNet50V2, but should be omitted for models like EfficientNetB0 
            that include their own rescaling. Default is False.
        name (str): Name of the Sequential model. Default is "data_augmentation".
            
    Returns:
        A keras.Sequential instance representing the data augmentation model.
    """
    data_augmentation_layers = [
        layers.RandomFlip(flip_mode),
        layers.RandomRotation(rotation_range),
        layers.RandomZoom(zoom_range),
        layers.RandomHeight(height_range),
        layers.RandomWidth(width_range),
    ]
    if include_rescaling:
        data_augmentation_layers.append(layers.Rescaling(1./255))

    data_augmentation = Sequential(data_augmentation_layers, name=name)

    return data_augmentation

  
  
def unfreeze_model_layers(model='base_model', num_last_layers_trainable=10):
    """
    Unfreeze the last few layers of a model.
    
    Args:
        model (str, optional): The name of the model whose layers are to be unfrozen. Default is 'base_model'.
        num_last_layers_trainable (int, optional): The number of last layers to be made trainable. Default is 10.
    
    Returns:
        None
    """
    # Fetch the model from global scope
    model = globals()[model]

    # Make all layers trainable
    model.trainable = True

    # Freeze all layers except for the last num_last_layers_trainable layers
    for layer in model.layers[:-num_last_layers_trainable]:
        layer.trainable = False

    print(f"The last {num_last_layers_trainable} layers of {model.name} are now trainable.")
  
  
  
import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15, norm=False, savefig=False, x_rotation=0): 
  """
  Makes a labelled confusion matrix comparing predictions and ground truth labels.

  Args:
    y_true: Array of truth labels (must be same shape as y_pred).
    y_pred: Array of predicted labels (must be same shape as y_true).
    classes: Array of class labels (e.g. string form). If `None`, integer labels are used.
    figsize: Size of output figure (default=(10, 10)).
    text_size: Size of output figure text (default=15).
    norm: normalize values or not (default=False).
    savefig: save confusion matrix to file (default=False).
    x_rotation: rotation angle for xticks (default=0).
  
  Returns:
    A labelled confusion matrix plot comparing y_true and y_pred.
  """  
  cm = confusion_matrix(y_true, y_pred)
  cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
  n_classes = cm.shape[0]

  fig, ax = plt.subplots(figsize=figsize)
  cax = ax.matshow(cm, cmap=plt.cm.Blues)
  fig.colorbar(cax)

  if classes:
    labels = classes
  else:
    labels = np.arange(cm.shape[0])
  
  ax.set(title="Confusion Matrix",
         xlabel="Predicted label",
         ylabel="True label",
         xticks=np.arange(n_classes),
         yticks=np.arange(n_classes), 
         xticklabels=labels, 
         yticklabels=labels)

  ax.xaxis.set_label_position("bottom")
  ax.xaxis.tick_bottom()

  threshold = (cm.max() + cm.min()) / 2.

  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    if norm:
      plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
              horizontalalignment="center",
              color="white" if cm[i, j] > threshold else "black",
              size=text_size)
    else:
      plt.text(j, i, f"{cm[i, j]}",
              horizontalalignment="center",
              color="white" if cm[i, j] > threshold else "black",
              size=text_size)

  # Adjusting the rotation of x-axis labels
  plt.xticks(rotation=x_rotation, fontsize=text_size)
  plt.yticks(fontsize=text_size)
  
  if savefig:
    fig.savefig("confusion_matrix.png")

    
import os
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def load_and_prep_image(filename, img_shape=224, scale=True):
    """
    Reads an image from filename, turns it into a tensor and reshapes into
    (224, 224, 3).

    Parameters
    ----------
    filename (str): string filename of target image
    img_shape (int): size to resize target image to, default 224
    scale (bool): whether to scale pixel values to range(0, 1), default True
    """
    # Read in the image
    img = tf.io.read_file(filename)
    # Decode it into a tensor
    img = tf.image.decode_image(img, channels=3)
    # Resize the image
    img = tf.image.resize(img, [img_shape, img_shape])
    # Rescale the image (get all values between 0 and 1) if needed
    return img/255. if scale else img

def predict_and_plot(model, test_dir, class_names, img_shape=224, num_images=3, figsize=(17, 10), scale=True):
    """
    Load random images, make predictions on them, and plot them.

    Parameters
    ----------
    model: Trained model.
    test_dir (str): Directory where test images are stored in class-name subdirectories.
    class_names (list): List of class names.
    img_shape (int): Target shape of image.
    num_images (int): Number of images to predict and plot, default 3.
    figsize (tuple): Figure size, default (17, 10).
    scale (bool): Whether to scale image pixel values to range(0, 1), default True.
    """
    # Set up figure
    plt.figure(figsize=figsize)

    for i in range(num_images):
        # Choose a random class and a random image from that class
        class_name = random.choice(class_names)
        filepath = random.choice(os.listdir(test_dir + "/" + class_name))

        # Load the image and make predictions
        img = load_and_prep_image(test_dir + "/" + class_name + "/" + filepath, img_shape=img_shape, scale=scale)
        pred = model.predict(tf.expand_dims(img, axis=0))

        # Get the predicted class
        if len(pred[0]) > 1: # multi-class
            pred_class = class_names[np.argmax(pred)]
        else: # binary class
            pred_class = class_names[int(tf.round(pred))]

        # Normalize img to [0, 1] for visualization
        img = img / np.max(img)

        # Plot the image and its prediction
        plt.subplot(1, num_images, i+1)
        plt.imshow(img)
        plt.title(f"actual: {class_name}, pred: {pred_class}, prob: {np.max(pred):.2f}", color=("green" if class_name==pred_class else "red"))
        plt.axis("off")
        
    plt.show()

