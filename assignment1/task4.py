import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import tqdm
import numpy as np
import utils
import dataloaders
import torchvision
from trainer import Trainer

torch.random.manual_seed(0)
np.random.seed(0)

# Load the dataset and print some stats
batch_size = 64

image_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(0.5, 0.5) # Task 4 a
])

dataloader_train, dataloader_test = dataloaders.load_dataset(
    batch_size, image_transform)
example_images, _ = next(iter(dataloader_train))
print(f"The tensor containing the images has shape: {example_images.shape} (batch size, number of color channels, height, width)",
      f"The maximum value in the image is {example_images.max()}, minimum: {example_images.min()}", sep="\n\t")


def create_model(*args: nn.Module):
    """
        Initializes the mode. Edit the code below if you would like to change the model.
    """
    model = nn.Sequential(
        *args # move for task 4 d
        # No need to include softmax, as this is already combined in the loss function
    )
    # Transfer model to GPU memory if a GPU is available
    model = utils.to_cuda(model)
    return model


# create model with these models
model = create_model(
    nn.Flatten(),  # Flattens the image from shape (batch_size, C, Height, width) to (batch_size, C*height*width)
    nn.Linear(28*28*1, 64), # introduce a hidden layer with linear transformation nodes
    nn.ReLU(),    
    nn.Linear(64, 10),
)


# Test if the model is able to do a single forward pass
example_images = utils.to_cuda(example_images)
output = model(example_images)
print("Output shape:", output.shape)
expected_shape = (batch_size, 10)  # 10 since mnist has 10 different classes
assert output.shape == expected_shape,    f"Expected shape: {expected_shape}, but got: {output.shape}"


# Hyperparameters
learning_rate = .0192
num_epochs = 5

# Use CrossEntropyLoss for multi-class classification
loss_function = torch.nn.CrossEntropyLoss()

# Define optimizer (Stochastic Gradient Descent)
optimizer = torch.optim.SGD(model.parameters(),
                            lr=learning_rate)


trainer = Trainer(
    model=model,
    dataloader_train=dataloader_train,
    dataloader_test=dataloader_test,
    batch_size=batch_size,
    loss_function=loss_function,
    optimizer=optimizer
)
train_loss_dict, test_loss_dict = trainer.train(num_epochs)


# We can now plot the training loss with our utility script

# Task 4 b
# source: https://matplotlib.org/3.1.1/gallery/axes_grid1/simple_axesgrid.html
if (False):
    # Note: it's more interesting to look at the old model as there you can actually see numbers :P
    grid = ImageGrid(plt.figure(), 
                    111,  
                    nrows_ncols=(2, 5), 
                    axes_pad=0.2 # padding in inches
                    )

    # Grab the Linear object out of the model
    linear = list(model.children())[1]

    # Grab weights for each class node 
    weights = linear.weight.cpu().data

    for ax, weight in zip(grid, weights):
        # Iterating over the grid returns the Axes.
        im = weight.reshape(28, 28)
        ax.imshow(im, cmap="gray")

    plt.savefig("image_solutions/class_weights_task4b.png")
    plt.show()
# End of task 4 b code



torch.save(model.state_dict(), "saved_model.torch")
final_loss, final_acc = utils.compute_loss_and_accuracy(
    dataloader_test, model, loss_function)
print(f"Final Test loss: {final_loss}. Final Test accuracy: {final_acc}")


# You can delete the remaining code of this notebook, as this is just to illustrate one method to solve the assignment tasks.


# This example code is here to illustrate how you can plot two different models to compare them.
# Lets change a small part of our model: the number of epochs trained (NOTE, use 5 epochs for your experiments in the assignment.)

# Task 4 d: do it the lazy way and use the code that was already here, altough we ligthly modify it

# We reset the manual seed to 0, such that the model parameters are initialized with the same random number generator.
torch.random.manual_seed(0)
np.random.seed(0)


# dataloader_train, dataloader_test = dataloaders.load_dataset(
#     batch_size, image_transform)

# use original models 
model = create_model(
    nn.Flatten(),  
    nn.Linear(28*28*1, 10), 
)

# Redefine optimizer, as we have a new model.
optimizer = torch.optim.SGD(model.parameters(),
                            lr=learning_rate)
trainer = Trainer(
    model=model,
    dataloader_train=dataloader_train,
    dataloader_test=dataloader_test,
    batch_size=batch_size,
    loss_function=loss_function,
    optimizer=optimizer
)
train_old_loss_dict_epochs, test_old_loss_dict_epochs = trainer.train(num_epochs)
num_epochs = 1


# We can now plot the two models against eachother

# Plot loss
utils.plot_loss(train_old_loss_dict_epochs, label="Train Loss - task 4 a model")
utils.plot_loss(test_old_loss_dict_epochs,  label="Test Loss - task 4 a model")
utils.plot_loss(train_loss_dict, label="Train Loss - 64 node hidden & relu")
utils.plot_loss(test_loss_dict, label="Test Loss - 64 node hidden & relu")
# Limit the y-axis of the plot (The range should not be increased!)
plt.ylim([0, 1])
plt.legend()
plt.xlabel("Global Training Step")
plt.ylabel("Cross Entropy Loss")
plt.savefig("image_solutions/task_4d.png")

plt.show()

torch.save(model.state_dict(), "saved_model.torch")
final_loss, final_acc = utils.compute_loss_and_accuracy(
    dataloader_test, model, loss_function)
print(f"Final Test loss: {final_loss}. Final Test accuracy: {final_acc}")
