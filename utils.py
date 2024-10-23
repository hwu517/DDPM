import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_score

def plot_loss_curve(losses):
    plt.plot(losses)
    plt.title('DDPM Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

def plot_denoising_steps(images):
    fig, axes = plt.subplots(1, len(images))
    for idx, img in enumerate(images):
        axes[idx].imshow(img, cmap='gray')
        axes[idx].set_title(f'Step {idx*10}')
    plt.show()

def calculate_precision_recall(y_true, y_pred):
    precision, recall = precision_recall_score(y_true, y_pred, average='macro')
    return precision, recall
