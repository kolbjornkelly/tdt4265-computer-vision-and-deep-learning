
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
from torchvision.utils import save_image
import torch
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid

image = Image.open("images/zebra.jpg")
print("Image shape:", image.size)

model = torchvision.models.resnet18(pretrained=True)
print(model)
first_conv_layer = model.conv1
print("First conv layer weight shape:", first_conv_layer.weight.shape)
print("First conv layer:", first_conv_layer)

# Resize, and normalize the image with the mean and standard deviation
image_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
image = image_transform(image)[None]
print("Image shape:", image.shape)

activation = first_conv_layer(image)
print("Activation shape:", activation.shape)


def torch_image_to_numpy(image: torch.Tensor):
    """
    Function to transform a pytorch tensor to numpy image
    Args:
        image: shape=[3, height, width]
    Returns:
        iamge: shape=[height, width, 3] in the range [0, 1]
    """
    # Normalize to [0 - 1.0]
    image = image.detach().cpu()  # Transform image to CPU memory (if on GPU VRAM)
    image = image - image.min()
    image = image / image.max()
    image = image.numpy()
    if len(image.shape) == 2:  # Grayscale image, can just return
        return image
    assert image.shape[0] == 3, "Expected color channel to be on first axis. Got: {}".format(
        image.shape)
    image = np.moveaxis(image, 0, 2)
    return image


indices = [14, 26, 32, 49, 52]

# Task 4b)


# Weight images

layer1_weights = []
layer1_fig = plt.figure()
grid = ImageGrid(layer1_fig, 111,
                 nrows_ncols=(1, 5),
                 axes_pad=0.1,
                 )
for i in indices:
    layer1_weights.append(torch_image_to_numpy(first_conv_layer.weight[i]))
for ax, img in zip(grid, layer1_weights):
    ax.imshow(img)
plt.show()

# Activation images
torchvision.utils.save_image(
    activation[0][14], "images/4b/activations14.png")
torchvision.utils.save_image(
    activation[0][26], "images/4b/activations26.png")
torchvision.utils.save_image(
    activation[0][32], "images/4b/activations32.png")
torchvision.utils.save_image(
    activation[0][49], "images/4b/activations49.png")
torchvision.utils.save_image(
    activation[0][52], "images/4b/activations52.png")


# Task 4c)

path = f"images/4c"
layer_count = 1
layer_weights = []
last_index = activation.shape[1] - 1
torchvision.utils.save_image(
    activation[0][last_index], f"{path}/layer{layer_count}.png")

for layer in model.children():
    if layer_count == 1:
        # First layer is allready passed
        layer_count += 1
    elif layer_count <= 8:
        activation = layer(activation)
        last_index = activation.shape[1] - 1
        torchvision.utils.save_image(
            activation[0][last_index], f"{path}/layer{layer_count}.png")
        layer_count += 1
    else:
        break
