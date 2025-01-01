import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.random.manual_seed(42) # We need to put this option in a config file, so that we can change it easily (also in the machine_learning_testing.py file)

# Let's create a random data generator, where the data generated has a linear relationship, and can be represented via a scatter plot.

def generate_data(n=100, noise=0.1):
    """ Used to generate random data """
    x = torch.rand(n)
    y = 2*x + 1 + noise*torch.randn(n)
    return x, y

# Used to display the data:
def show_data(x, y):
    """ Used to display the data """
    plt.scatter(x, y)
    plt.xlabel('sugar content (example)')
    plt.ylabel('calories (example)')
    plt.title('Generated data')
    plt.show()

if __name__ == '__main__':
    # We provide an example of the data that is being generated...
    x, y = generate_data()
    show_data(x, y)
