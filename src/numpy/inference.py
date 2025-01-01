import os
import numpy as np
import matplotlib.pyplot as plt

from train import Linear 
from train import n_samples, data_noise
from data import generate_data

# This is an inference script for our scratch model that we made in the train.py script
# in the same directory

model = Linear()

current_script_path = os.path.dirname(os.path.abspath(__file__))
# Let's load the model that we trained using numpy

# Set the two parameters equal to the ones we trained
model.w, model.b = np.load(os.path.join(current_script_path, "../../models/model.npy"), allow_pickle=True, encoding='latin1')

data = generate_data(n_samples, noise=data_noise)
x, _ = data


def predict_value(prediction_input):
    """ Predicts the output (dimension 1) of the model given an input tensor. """
    # We retrieve the outputs:
    output = model.forward(prediction_input)

    return output


def get_predictions(x):
    """ Iterates over data x and makes predictions to append to a list. """
    predictions = []
    for i in range(len(x)):
        input = x[i]
        output = predict_value(input)
        predictions.append(output)
    return predictions


def plot_predictions_and_actual(data):
    """ Plots the actual data and the predictions made by the model. """
    # NOTE: This function code was actually taken (and slightly modified) from one of the resources in the README: [Simple Linear Regression](https://www.kaggle.com/code/devzohaib/simple-linear-regression). Thanks to the author!
    x, y = data
    predictions = get_predictions(x)

    c = list(range(0,len(x),1)) # generating index 
    fig = plt.figure()
    plt.plot(c, y, color="blue", linewidth=2, linestyle="-")
    plt.plot(c, predictions, color="red",  linewidth=2, linestyle="-")
    fig.suptitle('Actual and Predicted', fontsize=20)              # Plot heading 
    plt.xlabel('Index', fontsize=18)                               # X-label
    plt.ylabel('calories (example)', fontsize=16)  
    plt.show()

# Let's create a function that iterates over the indpendent variable, and makes predictions, so that we can plot the line that the model predicts:
def plot_model_predictions_regression(data):
    """ Plots the linear regression line that the model predicts based on its predictions over all the data """
    x, y = data
    predictions = get_predictions(x)
    plt.scatter(x, y)
    plt.plot(x, predictions, color='red')
    plt.xlabel('sugar content (example)')
    plt.ylabel('calories (example)')
    plt.title('Model Predictions')
    plt.show()
    
# Let's now create a function that gets the slope and the intercept of the model line based on the predictions:
def get_model_regression_line(data):
    """ Computes the slope and the intercept of the regression line based on the model predictions over the data. """
    x, y = data
    predictions = get_predictions(x)
    x_min = np.min(x)
    x_max = np.max(x)
    y_min = min(predictions) 
    y_max = max(predictions) 
    slope = (y_max - y_min) / (x_max - x_min)
    intercept = y_min - slope * x_min 
    return slope, intercept

# We can get the actual regression line via the least squares method:
def get_statistical_regression_line(data):
    """ Computes the slope and the intercept of the regression line based on the data."""
    x, y = data # Every linear regression line passed through the mean of the data for both x and y.
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_diff = x - x_mean
    y_diff = y - y_mean
    slope = np.sum(x_diff * y_diff) / np.sum(x_diff**2)
    intercept = y_mean - slope * x_mean
    return slope, intercept


def plot_statistical_regression_line(data):
    """ Plots the statistical linear regression line based on its predictions over all the data """
    x, y = data
    slope, intercept = get_statistical_regression_line(data)
    x_min = np.min(x)
    x_max = np.max(x)
    y_min = slope * x_min + intercept
    y_max = slope * x_max + intercept
    plt.scatter(x, y)
    plt.plot([x_min, x_max], [y_min, y_max], color='red')
    plt.xlabel('sugar content (example)')
    plt.ylabel('calories (example)')
    plt.title('Statistical Regression Line')
    plt.show()

if __name__ == '__main__':
    # We get the slope and the intercept of the model line:
    slope, intercept = get_model_regression_line(data)
    print("Model's Predicted Line:")
    print(f'Predicted Line: y = {slope}x + {intercept}')
    print(f'Slope: {slope}, Intercept: {intercept}\n')
    # We can now see the data, along with the plotted line that the model predicts:
    plot_model_predictions_regression(data)
    
    # We can also get the slope and the intercept of the actual regression line using the least squares method:
    slope, intercept = get_statistical_regression_line(data)
    print("Statistical Predicted Line (Least squares method):")
    print(f'Predicted Line: y = {slope}x + {intercept}')
    print(f'Slope: {slope}, Intercept: {intercept}')
    plot_statistical_regression_line(data)
    
    # We can also plot the actual and predicted values to inspect the model's performance:
    plot_predictions_and_actual(data)
