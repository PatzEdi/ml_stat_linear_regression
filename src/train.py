import torch
import torch.nn as nn
from torch import functional as F
import random
from data import generate_data
import matplotlib.pyplot as plt

use_random_seed = False

random_seed = random.randint(0, 1000) if use_random_seed else 42
print(f'Random seed: {random_seed}')
# We set the seed to make the results reproducible: 
torch.random.manual_seed(random_seed)

# Function to count the number of parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# We get the data:
x, y = generate_data(noise=0.3)

# Let's go ahead and define our model. We will use a simple linear model, which is a linear function that takes an input and produces an output.

model = nn.Linear(1, 1) # We don't need to use non-linear activation funcs, as the data is linear.
# We can now define our loss function. We will use the Mean Squared Error (MSE) loss function, which is a common loss function used in regression problems (similar to least squares method)
criterion = nn.MSELoss()
# We can now define our optimizer. We will use the Stochastic Gradient Descent (SGD) optimizer, which is a common optimizer used in linear regression problems. This is because we want each batch to not represent the entire dataset, but rather a random sample of it to avoid overfitting.
optimizer = torch.optim.SGD(model.parameters(), lr=0.0005)
# We can now train our model. We will train it for 100 epochs, which is a common number of epochs used in linear regression problems.
num_epochs = 1000 # This should be enough considering the learning rate...
# train loop...
model.train()
if __name__ == '__main__':
    losses = [] # We use this to append losses for later use and analyzing... 
    for epoch in range(num_epochs):
        for i in range(len(x)): # We iterate over our data set:
            input = x[i].unsqueeze(0)
            target = y[i].unsqueeze(0)
            output = model(input)
            # We compute the loss...
            loss = criterion(output, target)
            optimizer.zero_grad() # We need to zero the gradients before computing the gradients so that the gradients are not accumulated from previous iterations.
            loss.backward()
            optimizer.step() # We update the parameters of the model using the gradients computed in the backward pass.
        
        losses.append(loss.item())

        # # We get the average loss if needed:
        # average_loss = sum(losses)/len(x) 
        print("Epoch: {}/{} Loss: {:.5f}".format(epoch+1, num_epochs, loss.item()))

    # Let's save the model:
    torch.save(model.state_dict(), 'models/model.pth')
    
    print("Training complete!")
    
    # We plot the losses:
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()
    