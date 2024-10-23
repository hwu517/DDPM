import tensorflow as tf
from ddpm_model import Unet
from classifier import create_classifier
from utils import plot_loss_curve, plot_denoising_steps, calculate_precision_recall

def train_ddpm_and_classifier(train_dataset, val_dataset, epochs=100, classifier_epochs=10):
    # 假设第一个 batch 的图像用于确定输入形状
    for img_batch, label_batch in train_dataset.take(1):
        input_shape = img_batch.shape[1:]  # 动态获取输入形状 (e.g., (28, 28, 1) 或 (32, 32, 3))
        break

    # 创建 DDPM 和分类器模型
    ddpm = Unet(input_shape=input_shape)  # 传递动态输入形状
    classifier = create_classifier(input_shape=(128,), num_classes=10)  # 保持分类器输入形状不变

    # 保存每个 epoch 的损失值和中间图像结果
    losses = []
    images_at_steps = []

    # 训练 DDPM 模型
    for epoch in range(epochs):
        epoch_loss = 0
        for step, (images, labels) in enumerate(train_dataset):
            # 执行正向过程 (denoise and get latent features)
            # 从反向过程还原图像
            # 在每个 epoch 中计算损失并收集可视化步骤
            latent_features = ddpm(images)
            loss = tf.reduce_mean(tf.square(latent_features - images))  # 计算损失（这里作为示例）

            epoch_loss += loss.numpy()
            losses.append(epoch_loss)

            if step % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Step {step}, Loss: {loss.numpy()}")

        # 可视化损失曲线和去噪步骤
        plot_loss_curve(losses)
        plot_denoising_steps(images_at_steps)

        # 在训练结束后提取潜在特征并进行分类
        latent_features = extract_latent_features(ddpm, train_dataset)  # 从 DDPM 提取特征
        classifier.fit(latent_features, label_batch, validation_data=(val_latents, val_labels), epochs=classifier_epochs)

    return ddpm, classifier

def extract_latent_features(ddpm, dataset):
    latents = []
    labels = []
    for images, labels_batch in dataset:
        latent_features = ddpm(images)
        latents.append(latent_features)
        labels.append(labels_batch)
    return tf.concat(latents, axis=0), tf.concat(labels, axis=0)

