import torch
import machine_learning_testing as mlt
from data import show_data
import matplotlib.pyplot as plt

model = mlt.model
data = mlt.x, mlt.y # Yes, this data is different from when the model we are inferencing was trained. However, the general pattern is the same, so we can still use it to see if the line the model predicts generally fits the data well.

# We load the pth file:
model.load_state_dict(torch.load('model.pth'))

def predict_value(prediction_input):
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
    predictions = []
    for i in range(len(x)):
        input = x[i].unsqueeze(0)
        output = predict_value(input)
        predictions.append(output.item())
    return predictions

def plot_predictions_and_actual(data):
    x, y = data

    predictions = get_predictions(x)

    c = [i for i in range(0,100,1)]         # generating index 
    fig = plt.figure()
    plt.plot(c, y, color="blue", linewidth=2, linestyle="-")
    plt.plot(c, predictions, color="red",  linewidth=2, linestyle="-")
    fig.suptitle('Actual and Predicted', fontsize=20)              # Plot heading 
    plt.xlabel('Index', fontsize=18)                               # X-label
    plt.ylabel('Sales', fontsize=16)  
    plt.show()
# Let's create a function that iterates over the indpendent variable, and makes predictions, so that we can plot the line that the model predicts:
def plot_model_predictions_regression(model, data):
    x, y = data
    predictions = get_predictions(x)
    plt.scatter(x, y)
    plt.plot(x, predictions, color='red')
    plt.xlabel('sugar content (example)')
    plt.ylabel('calories (example)')
    plt.title('Model Predictions')
    plt.show()

if __name__ == '__main__':
    # We can now see the data, along with the plotted line that the model predicts:
    plot_model_predictions_regression(model, data)
    plot_predictions_and_actual(data)