
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
from torchvision.utils import save_image
import torch
import numpy as np
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

# TODO: Gjør dette i loop og få alle bilder i ett grid
# NB: må lage images/4b først

# TODO: spør studass om disse bildene gir mening (særlig weights)

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


img = torch_image_to_numpy(first_conv_layer.weight[14])
plt.imsave("images/4b/weights14.png", img)

img = torch_image_to_numpy(first_conv_layer.weight[26])
plt.imsave("images/4b/weights26.png", img)

img = torch_image_to_numpy(first_conv_layer.weight[32])
plt.imsave("images/4b/weights32.png", img)

img = torch_image_to_numpy(first_conv_layer.weight[49])
plt.imsave("images/4b/weights49.png", img)

img = torch_image_to_numpy(first_conv_layer.weight[52])
plt.imsave("images/4b/weights52.png", img)


# Task 4c)
# TODO: spør studass om man skal visualiere alle lagene, eller bare det siste
# TODO: Finner bare 10 lag totalt - skal alle med, eller skal
#       man fortsatt droppe de to siste?
layer_count = 1

for layer in model.children():
    if layer_count == 1:
        # First layer is allready passed
        layer_count += 1
    elif layer_count <= 8:
        activation = layer(activation)
        layer_count += 1
    else:
        break

torchvision.utils.save_image(
    activation[0][511], "images/4c/activations.png")
print("Activation_ten shape:", activation.shape)
