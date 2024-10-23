import tensorflow as tf
from ddpm_model import Unet
from classifier import create_classifier
from utils import plot_loss_curve, plot_denoising_steps, calculate_precision_recall, calculate_loss

def train_ddpm_and_classifier(dataset, val_dataset, epochs=100, classifier_epochs=10):
    ddpm = Unet()
    classifier = create_classifier(input_shape=(128,), num_classes=10)
    
    losses = []  # To store loss at each epoch
    for epoch in range(epochs):
        latent_features_list = []
        labels_list = []
        
        # Training DDPM
        for batch_images, batch_labels in dataset:
            # Forward process: denoise and get latent features
            latent_features = ddpm(batch_images, time=tf.random.uniform([batch_images.shape[0]]))

            # Reverse process: reconstruct images from noise
            reconstructed_images = ddpm(latent_features, time=tf.random.uniform([batch_images.shape[0]]))
            
            # Calculate loss
            loss = calculate_loss(reconstructed_images, batch_images)
            losses.append(loss)

            # Store latent features and labels for classifier training later
            latent_features_list.append(latent_features)
            labels_list.append(batch_labels)
        
        # Visualize loss curve and denoising steps for the current epoch
        plot_loss_curve(losses)
        plot_denoising_steps(batch_images, reconstructed_images)

        # After extracting latent features, train the classifier
        latents = tf.concat(latent_features_list, axis=0)
        labels = tf.concat(labels_list, axis=0)
        classifier.fit(latents, labels, validation_data=val_dataset, epochs=classifier_epochs)
    
    return ddpm, classifier
