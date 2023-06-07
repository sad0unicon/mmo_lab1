import os
import time
import psutil
import torch
import tensorflow as tf
from PIL import Image
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
#import onnxruntime
import numpy as np

def predict_images():
    # Define the paths to the directories containing the images
    images_dir1 = 'C:/Users/Owner/Documents/магистратура/РНС/ПредДав/Капибара'
    images_dir2 = 'C:/Users/Owner/Documents/магистратура/РНС/ПредДав/Кенгуру'
    images_dir3 = 'C:/Users/Owner/Documents/магистратура/РНС/ПредДав/Медведь'


    # Загрузка ONNX модели
    sess = torch.jit.load('cifar_cnn_fine.pt') # путь до модели
    #sess.eval()

    # Определение входных и выходных имен модели ONNX
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    # Определите список файлов изображений для обработки
    image_files1 = os.listdir(images_dir1)[:100]
    image_files2 = os.listdir(images_dir2)[:100]
    image_files3 = os.listdir(images_dir3)[:100]
    image_files = image_files1 + image_files2 + image_files3
    print(len(image_files))
    # Прокрутите файлы изображений и загрузите их
    images = []
    for image_file in image_files:
        image_path = os.path.join(images_dir1, image_file)
        if not os.path.exists(image_path):
            image_path = os.path.join(images_dir2, image_file)
        if not os.path.exists(image_path):
            image_path = os.path.join(images_dir3, image_file)
        image = Image.open(image_path).convert('RGB')
        image = np.asarray(image.resize((32, 32), Image.ANTIALIAS))
        images.append(image)

    # Define the list to store the predictions
    predictions = []
    # Define the start time
    start_time = time.time()

    # Loop through the images and run the ONNX model on them
    for image in images:
        output = sess.run([output_name], {input_name: np.asarray([image]).astype(np.float32)})
        prediction = np.argmax(output)
        predictions.append(prediction)

    # Define the end time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Calculate the memory usage
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / 1024 / 1024

    # Return the predictions, elapsed time, and memory usage
    return predictions, elapsed_time, memory_usage

predictions, elapsed_time, memory_usage = predict_images()
print("Elapsed time: ", elapsed_time)
print("Memory usage: ", memory_usage, "MB")