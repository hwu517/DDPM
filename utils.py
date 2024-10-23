import matplotlib.pyplot as plt
import tensorflow as tf

def plot_loss_curve(losses):
    plt.figure()
    plt.plot(losses)
    plt.title('DDPM Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

def plot_denoising_steps(original_images, denoised_images):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(original_images[0], cmap='gray')
    axes[0].set_title('Original Image')
    
    axes[1].imshow(denoised_images[0], cmap='gray')
    axes[1].set_title('Denoised at Step 10')
    
    axes[2].imshow(denoised_images[10], cmap='gray')
    axes[2].set_title('Denoised at Step 50')
    
    plt.show()

def calculate_loss(reconstructed_images, original_images):
    return tf.reduce_mean(tf.losses.mean_squared_error(reconstructed_images, original_images))

def calculate_precision_recall(y_true, y_pred):
    precision = tf.keras.metrics.Precision()
    recall = tf.keras.metrics.Recall()
    precision.update_state(y_true, y_pred)
    recall.update_state(y_true, y_pred)
    return precision.result().numpy(), recall.result().numpy()
