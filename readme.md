# Android Layout Machine Learning 

An attempt at using machine learning to determine whether a given Android xml layout has a send button. Uses binary classification so it can only determine the presence of a send button and no other details.

The dataset was created by dumping various application layouts using 'uiautomator'. 

Accuracy is terrible (44% and even that is most likely a fluke).

## Installation / Execution
Requires Python 3.6 - tensorflow was not working on 3.7

Install tensorflow and matplotlib using pip 
