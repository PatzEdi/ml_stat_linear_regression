import os
import numpy as np
import matplotlib.pyplot as plt
from data import generate_data 

use_random_seed = False 
seed = 42 if not use_random_seed else np.random.randint(0, 1000)
print(f"Using seed: {seed}")
# Let's make sure we can set a random seed so that the results are repoducible
np.random.seed(seed)

# Let's start defining our compoonents:

# We need a simple linear transformation, similar to nn.Linear(1,1). So, it will
# look something like: f(x) = wx + b.
# We need to define the parameters w and b, and we need to define the forward
# function that will take an input x and return the output wx + b.
# We also need to define the backward function that will compute the gradients
# of the loss with respect to w and b.
# We also need to define the optimizer that will update the parameters w and b
# using the gradients computed in the backward pass.

class Linear:
    def __init__(self):
        # Random initialization of the parameters (weights and biases)
        self.w = np.random.randn()
        self.b = np.random.randn()
    
    def forward(self, x):
        # Forward pass returns the output of the linear transformation
        return self.w*x + self.b
    
    def backward(self, x, grad):
        # The backward pass computes gradients of loss w.r.t. parameters
        # grad*x gives gradient for weight w (chain rule: dL/dw = dL/dy * dy/dw = grad * x) 
        self.w_grad = grad*x
        # grad alone is gradient for bias b (chain rule: dL/db = dL/dy * dy/db = grad * 1)
        self.b_grad = grad
        # Return gradients for use in parameter updates. These are with respect to the loss
        return self.w_grad, self.b_grad

# Define our loss
class MeanSquaredError:
    def forward(self, y_pred, y_true):
        # Forward pass computes the loss
        return np.mean((y_pred - y_true)**2)
    
    def backward(self, y_pred, y_true):
        # Backward pass computes the gradient of the loss w.r.t. the output
        return 2*(y_pred - y_true) # We don't have more than 1 sample per pass, so we don't need to divide by len(y_true)

# Define our optimizer
class SGD:
    def __init__(self, model, learning_rate):
        # Initialize the optimizer with the learning rate and the model
        self.learning_rate = learning_rate
        self.model = model 
    
    def step(self, w_grad, b_grad):
        # Update the parameters using the gradients
        self.model.w -= self.learning_rate*w_grad
        self.model.b -= self.learning_rate*b_grad
    
    def zero_grad(self):
        # Zero the gradients. For this, we just set them to zero
        self.model.w_grad = 0
        self.model.b_grad = 0
 
# Let's define some hyperparameters
n_samples = 100
data_noise = 0.3
n_epochs = 1000 # If you want it to be closer and closer to the least-squares solution, you need to increase the number of epochs
learning_rate = 0.0005


class Train:
    @staticmethod # No need for self, as we are not using instance variables
    def train(n_samples, n_epochs, learning_rate):
        
        model = Linear()
        
        criterion = MeanSquaredError()
        
        optimizer = SGD(model, learning_rate) 

        x, y = generate_data(n_samples, noise=data_noise, use_random_seed=False) # Data
        
        losses = []
        for epoch in range(n_epochs):
            total_loss = 0
            for i in range(len(x)):
                # We get our inputs and targets
                input = x[i]
                target = y[i]
                
                # Forward pass
                output = model.forward(input) # model forward
                # We calculate the loss:
                loss = criterion.forward(output, target) # loss forward
                total_loss += loss

                # We calculate the gradients that the model needs to update the parameters:
                grad = criterion.backward(output, target) # loss backward
                w_grad, b_grad = model.backward(input, grad)
                
                # We update the parameters:
                optimizer.step(w_grad, b_grad)
                # We zero the gradients:
                optimizer.zero_grad()
                
            # Average loss reporting
            average_epoch_loss = total_loss/len(x)
            losses.append(average_epoch_loss)
            print(f"Epoch: {epoch+1}/{n_epochs} Loss: {average_epoch_loss}") 

        print("Training complete!")

        return model, losses

if __name__ == "__main__":
    trained_model, losses = Train.train(n_samples, n_epochs, learning_rate)
    
    current_script_path = os.path.dirname(os.path.abspath(__file__))
    model_save_path = os.path.join(current_script_path, "../../models/model.npy")
    # We can save the weights and biases of the trained model
    np.save(model_save_path, np.array([trained_model.w, trained_model.b]))
    print(f"\nModel saved at: {model_save_path}")
    
    # We plot the losses:
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()
