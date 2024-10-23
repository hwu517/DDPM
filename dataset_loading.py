import tensorflow as tf

def load_datasets(dataset_name='mnist', batch_size=64):
    if dataset_name == 'mnist':
        (train_images, train_labels), (val_images, val_labels) = tf.keras.datasets.mnist.load_data()
        train_images = train_images[..., tf.newaxis].astype("float32") / 255.0
        val_images = val_images[..., tf.newaxis].astype("float32") / 255.0

    elif dataset_name == 'fashion_mnist':
        (train_images, train_labels), (val_images, val_labels) = tf.keras.datasets.fashion_mnist.load_data()
        train_images = train_images[..., tf.newaxis].astype("float32") / 255.0
        val_images = val_images[..., tf.newaxis].astype("float32") / 255.0

    elif dataset_name == 'cifar10':
        (train_images, train_labels), (val_images, val_labels) = tf.keras.datasets.cifar10.load_data()
        train_images = train_images.astype("float32") / 255.0
        val_images = val_images.astype("float32") / 255.0

    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")

    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))

    train_dataset = train_dataset.shuffle(10000).batch(batch_size)
    val_dataset = val_dataset.batch(batch_size)

    return train_dataset, val_dataset
