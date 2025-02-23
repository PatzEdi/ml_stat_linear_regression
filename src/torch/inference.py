import os
import torch
import matplotlib.pyplot as plt
import train as mlt
from train import count_parameters

model = mlt.model
data = mlt.x, mlt.y # Yes, this data is different from when the model we are inferencing was trained. However, the general pattern is the same, so we can still use it to see if the line the model predicts generally fits the data well.
current_script_path = mlt.current_script_path
model_save_path = os.path.join(current_script_path, '../../models/model.pth')
# We load the pth file:
model.load_state_dict(torch.load(model_save_path))

num_parameters = count_parameters(model)
print(f'Number of parameters: {num_parameters}\n') # With a single linear layer, we have 2 parameters (weight and bias). It is very small, yet it can still be used to make predictions. The model size is 1 KB, which is very small.

model.eval()

def predict_value(prediction_input):
    """ Predicts the output (dimension 1) of the model given an input tensor. """
    with torch.no_grad():
        # We retrieve the outputs:
        output = model(prediction_input)

        return output

# We can now use the model to make predictions (example):
# input = 1.0 
# input = torch.tensor([input])
# output = predict_value(input)
# print(output)
def get_predictions(x):
    """ Iterates over data x and makes predictions to append to a list. """
    predictions = []
    for i in range(len(x)):
        input = x[i].unsqueeze(0)
        output = predict_value(input)
        predictions.append(output.item())
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
def plot_model_predictions_regression(model, data):
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
    x_min = torch.min(x)
    x_max = torch.max(x)
    y_min = min(predictions) 
    y_max = max(predictions) 
    slope = (y_max - y_min) / (x_max - x_min)
    intercept = y_min - slope * x_min 
    return slope, intercept

# We can get the actual regression line via the least squares method:
def get_statistical_regression_line(data):
    """ Computes the slope and the intercept of the regression line based on the data."""
    x, y = data # Every linear regression line passed through the mean of the data for both x and y.
    x_mean = torch.mean(x)
    y_mean = torch.mean(y)
    x_diff = x - x_mean
    y_diff = y - y_mean
    slope = torch.sum(x_diff * y_diff) / torch.sum(x_diff**2)
    intercept = y_mean - slope * x_mean
    return slope, intercept


def plot_statistical_regression_line(data):
    """ Plots the statistical linear regression line based on its predictions over all the data """
    x, y = data
    slope, intercept = get_statistical_regression_line(data)
    x_min = torch.min(x)
    x_max = torch.max(x)
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
    plot_model_predictions_regression(model, data)
    
    # We can also get the slope and the intercept of the actual regression line using the least squares method:
    slope, intercept = get_statistical_regression_line(data)
    print("Statistical Predicted Line (Least squares method):")
    print(f'Predicted Line: y = {slope}x + {intercept}')
    print(f'Slope: {slope}, Intercept: {intercept}')
    plot_statistical_regression_line(data)
    
    # We can also plot the actual and predicted values to inspect the model's performance:
    plot_predictions_and_actual(data)
