import tensorflow as tf
from ddpm_model import Unet
from classifier import create_classifier
from utils import plot_loss_curve, plot_denoising_steps, calculate_precision_recall

def train_ddpm_and_classifier(dataset, epochs=100):
    ddpm = Unet()
    classifier = create_classifier(input_shape=(128,), num_classes=10)

    # Train the DDPM model (implement the forward and reverse process)
    for epoch in range(epochs):
        for batch in dataset:
            # Forward process (denoise and get latent features)
            # Reverse process to reconstruct images
            # Save loss, visualize denoising steps

            plot_loss_curve(losses)  # Save loss curve after each epoch
            plot_denoising_steps(images_at_steps)  # Save denoising steps visualization

        # After extracting latent features
        classifier.fit(latents, labels, validation_data=(val_latents, val_labels), epochs=10)
    
    return ddpm, classifier
