import numpy as np
import matplotlib.pyplot as plt

# Let's create a random data generator, where the data generated has a linear relationship, and can be represented via a scatter plot.

def generate_data(n=100, noise=0.1, use_random_seed=False):
    """ Used to generate random data """
    # We set the seed
    seed = np.random.randint(0, 1000) if use_random_seed else 42
    np.random.seed(seed)
    
    x = np.random.rand(n).astype(np.float32)
    y = (2*x + 1 + noise*np.random.randn(n)).astype(np.float32)
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