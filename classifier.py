# Trevor Haigh
# The goal of this program is to use machine learning to determine if an android layout has a 'send' button
# It then shows a plot of the accuracy (it's bad)

import tensorflow as tf
from tensorflow import keras
import numpy
import re
import os
import ast
import matplotlib.pyplot as plt

dataset_dir = './dataset'
testdata_dir = './testdata'

def main():
    training_data, training_labels = read_layouts(dataset_dir)
    test_data, test_labels = read_layouts(testdata_dir)
    history, results = learning(training_data, training_labels, test_data, test_labels)
    print('Test Accuracy Results: ', results[1])
    plot_history(history)
    

# Iterates a given directory and parses the xml files contained within
# Returns two arrays: one of the data, and the other being the labels corresponding to that data
#       1 = has a send button, 0 = no send button
def read_layouts(dataset_dir):
    data = []
    labels = []
    for f in os.listdir(dataset_dir):
        if f.endswith('.xml'):
            with open(os.path.join(dataset_dir, f), 'r') as xml_file:
                for xml in xml_file:
                    parsed_data = parse_xml(xml)
            data.append(parsed_data)
    
            # labels the data. filenames have '1' in them if they have a send button. 
            # '2' is the same app with no send button
            if '1' in f:
                labels.append(1)
            else:
                labels.append(0)
        else:
            print(f, ' is not an xml file')

    return data, labels


# Parses xml content to retrieve data
# Uses the x,y and size of each node as data for input
# Each node has a 'bounds' attribute which is used for this data
def parse_xml(xml):
    data = []
    # Get all of the bounds values
    bounds_list = re.findall(r'(\[\d+,\d+\])', xml)
    for i in range(0, len(bounds_list), 2):
        # get x,y coords
        x,y = ast.literal_eval(bounds_list[i])
        # use next pair to calculate area
        a,b = ast.literal_eval(bounds_list[i+1])
        area = (a-x) * (b-y)
        data.append(x)
        data.append(y)
        data.append(area)

    return data


# Do the actual learning
# Probably a much better model to use, but I'm not sure.
def learning(training_data, training_labels, test_data, test_labels):
    training_data = keras.preprocessing.sequence.pad_sequences(training_data, value=1, padding='post', maxlen=1000)

    test_data = keras.preprocessing.sequence.pad_sequences(training_data, value=1, padding='post', maxlen=1000)

    model = keras.Sequential()
    model.add(keras.layers.Embedding(10000000, 16))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(16, activation=tf.nn.relu))
    model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

    model.compile(optimizer=tf.train.AdamOptimizer(), loss='binary_crossentropy', metrics=['accuracy'])

    # Split the training data/labels to use for training and validation in the learning phase
    partial_train = training_data[5:]
    partial_labels = training_labels[5:]
   
    validation_data = training_data[:4]
    validation_labels = training_labels[:4]

    history = model.fit(partial_train, partial_labels, epochs=30, batch_size=512, 
                        validation_data=(validation_data,validation_labels),
                        verbose=1)
    
    # Test the model against a data set it hasn't seen
    results = model.evaluate(test_data, test_labels)
    return history, results

# Create a graph of the accuracy over time
# The dots are the accuracy during the training
# The line is the accuracy using the test set
def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Accuracy over time')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.show()

if __name__ == '__main__':
    main()
