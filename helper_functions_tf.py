def plot_history(history):
  """
  Plots the training and validation loss and accuracy of a trained model.

  Args:
  history: History object. Outputs of the fit() function of a keras model.

  Returns:
  Two plots. One for loss and one for accuracy.
  """
  
  # Importing the necessary library for plotting
  import matplotlib.pyplot as plt
  
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


def download_and_unzip(filepath):
  """
  Downloads and unzips a zip file from a specified filepath.

  Args:
  filepath: A string specifying the URL of the zip file to download.

  Returns:
  None.
  """
  # Import necessary libraries
  import os
  import zipfile

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
  # Import libraries
  import matplotlib.pyplot as plt
  import matplotlib.image as mpimg
  import os
  import random

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

