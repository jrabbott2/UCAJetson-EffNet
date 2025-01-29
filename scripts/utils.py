import os
import torch
import matplotlib.pyplot as plt

def plot_loss(train_losses, test_losses, output_dir):
    """
    Plots training and testing losses over epochs and saves the plot as an image.
    """
    plt.figure()
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b--', label='Training Loss')
    plt.plot(epochs, test_losses, 'orange', label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.title('Training and Test Losses Over Epochs')

    # Save plot to the output directory
    plot_filename = os.path.join(output_dir, 'loss_plot.png')
    plt.savefig(plot_filename)
    print(f"Loss plot saved at {plot_filename}")


def save_model(model, output_dir, epoch, lr):
    """
    Saves the model state_dict to the output directory.
    """
    model_filename = os.path.join(output_dir, f'model_epoch_{epoch+1}_lr_{lr}.pth')
    torch.save(model.state_dict(), model_filename)
    print(f"Model saved at {model_filename}")
