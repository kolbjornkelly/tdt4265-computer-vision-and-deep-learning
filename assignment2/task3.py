import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images, one_hot_encode, SoftmaxModel
from task2 import SoftmaxTrainer


if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = .02
    batch_size = 32
    neurons_per_layer10 = [60, 60, 10]
    neurons_per_layer = [64, 10]
    momentum_gamma = 0.9  # Task 3 hyperparameter
    shuffle_data = True

    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_momentum = True

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)

    model10 = SoftmaxModel(
        neurons_per_layer10,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer10 = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model10, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history10, val_history10 = trainer10.train(num_epochs)

"""
plt.figure(figsize=(20, 12))
plt.subplot(1, 2, 1)
plt.ylim([0, .7])
utils.plot_loss(train_history["loss"],
                "Training", npoints_to_average=10)
utils.plot_loss(val_history["loss"],
                "Validation", npoints_to_average=10)
plt.legend()
plt.xlabel("Number of Training Steps")
plt.ylabel("Cross Entropy Loss")
# Plot accuracy
plt.subplot(1, 2, 2)
plt.ylim([.7, 1])
utils.plot_loss(train_history["accuracy"], "Training")
utils.plot_loss(val_history["accuracy"], "Validation")
plt.xlabel("Number of Training Steps")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("task4d.png")
plt.show()
"""

# Plot for 4e:

plt.figure(figsize=(20, 12))
plt.subplot(1, 2, 1)
plt.ylim([0, .7])
utils.plot_loss(train_history["loss"],
                "1 HL", npoints_to_average=10)
utils.plot_loss(train_history10["loss"],
                "10 HL", npoints_to_average=10)

plt.legend()
plt.xlabel("Number of Training Steps")
plt.ylabel("Training loss")
# Plot accuracy
plt.subplot(1, 2, 2)
plt.ylim([.7, 1])
utils.plot_loss(train_history["accuracy"], "Training, 1 HL")
utils.plot_loss(val_history["accuracy"], "Validation, 1 HL")
utils.plot_loss(train_history10["accuracy"], "Training, 10 HL")
utils.plot_loss(val_history10["accuracy"], "Validation, 10 HL")
plt.xlabel("Number of Training Steps")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("task4_good_grad.png")
plt.show()


# Everythin bellow is used to make plots for task 3
"""

# Compare with weight init
use_improved_weight_init = True
model_weight = SoftmaxModel(
    neurons_per_layer,
    use_improved_sigmoid,
    use_improved_weight_init)
trainer_weight = SoftmaxTrainer(
    momentum_gamma, use_momentum,
    model_weight, learning_rate, batch_size, shuffle_data,
    X_train, Y_train, X_val, Y_val,
)
train_history_weight, val_history_weight = trainer_weight.train(
    num_epochs)


plt.figure(figsize=(20, 12))
plt.subplot(1, 2, 1)
utils.plot_loss(train_history["loss"],
                "Task 2 Model", npoints_to_average=10)
utils.plot_loss(
    train_history_weight["loss"], "Improved Weight Initialization", npoints_to_average=10)
plt.ylim([0, .7])
plt.xlabel("Number of Training Steps")
plt.ylabel("Training Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.ylim([.7, 1])
utils.plot_loss(val_history["accuracy"], "Validation, no weight init")
utils.plot_loss(train_history["accuracy"], "Training, no weight init")
utils.plot_loss(
    val_history_weight["accuracy"], "Validation w/ weight init")
utils.plot_loss(
    train_history_weight["accuracy"], "Training w/ weight init")
plt.xlabel("Number of Training Steps")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("task3_a_weights.png")


# Compare with sigmoid

use_improved_sigmoid = True
model_sigmoid = SoftmaxModel(
    neurons_per_layer,
    use_improved_sigmoid,
    use_improved_weight_init)
trainer_sigmoid = SoftmaxTrainer(
    momentum_gamma, use_momentum,
    model_sigmoid, learning_rate, batch_size, shuffle_data,
    X_train, Y_train, X_val, Y_val,
)
train_history_sigmoid, val_history_sigmoid = trainer_sigmoid.train(
    num_epochs)


plt.figure(figsize=(20, 12))
plt.subplot(1, 2, 1)
utils.plot_loss(
    train_history_weight["loss"], "No improved sigmoid", npoints_to_average=10)
utils.plot_loss(
    train_history_sigmoid["loss"], "Improved Sigmoid", npoints_to_average=10)
plt.ylim([0, .7])
plt.xlabel("Number of Training Steps")
plt.ylabel("Training Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.ylim([.7, 1])
utils.plot_loss(
    val_history_weight["accuracy"], "Validation, no improved sigmoid")
utils.plot_loss(
    train_history_weight["accuracy"], "Training, no improved sigmoid")
utils.plot_loss(
    val_history_sigmoid["accuracy"], "Validation w/ improved sigmoid")
utils.plot_loss(
    train_history_sigmoid["accuracy"], "Training w/ improved sigmoid")
plt.xlabel("Number of Training Steps")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("task3b_sigmoid.png")


# Compare with momentum

use_momentum = True
learning_rate = .02

model_momentum = SoftmaxModel(
    neurons_per_layer,
    use_improved_sigmoid,
    use_improved_weight_init)
trainer_momentum = SoftmaxTrainer(
    momentum_gamma, use_momentum,
    model_momentum, learning_rate, batch_size, shuffle_data,
    X_train, Y_train, X_val, Y_val,
)
train_history_momentum, val_history_momentum = trainer_momentum.train(
    num_epochs)


plt.figure(figsize=(20, 12))
plt.subplot(1, 2, 1)
utils.plot_loss(
    train_history_sigmoid["loss"], "No Momentum", npoints_to_average=10)
utils.plot_loss(
    train_history_momentum["loss"], "With Momentum", npoints_to_average=10)
plt.ylim([0, .7])
plt.xlabel("Number of Training Steps")
plt.ylabel("Training Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.ylim([.7, 1])
utils.plot_loss(
    val_history_sigmoid["accuracy"], "Validation, no momentum")
utils.plot_loss(
    train_history_sigmoid["accuracy"], "Training, no momentum")

utils.plot_loss(
    val_history_momentum["accuracy"], "Validation w/ momentum")
utils.plot_loss(
    train_history_momentum["accuracy"], "Training w/ momentum")
plt.xlabel("Number of Training Steps")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("task3c_momentum.png")
"""
