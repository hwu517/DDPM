import tensorflow_datasets as tfds
from train import train_ddpm_and_classifier
from utils import calculate_precision_recall

# Load the MNIST, CIFAR-10, Fashion-MNIST datasets
dataset, info = tfds.load('mnist', with_info=True, as_supervised=True)
train_dataset = dataset['train'].batch(64)
test_dataset = dataset['test'].batch(64)

# Train DDPM and MLP Classifier
ddpm_model, classifier_model = train_ddpm_and_classifier(train_dataset)

# Evaluate the classifier
test_acc = classifier_model.evaluate(test_dataset)
print(f"Test Accuracy: {test_acc}")

# Calculate precision and recall
y_true = np.concatenate([y for _, y in test_dataset])
y_pred = classifier_model.predict(test_dataset)
precision, recall = calculate_precision_recall(y_true, y_pred)
print(f"Precision: {precision}, Recall: {recall}")
