import tensorflow as tf
import tensorflow.keras.layers as nn
from functools import partial

class SinusoidalPosEmb(nn.Layer):
    def __init__(self, dim, max_positions=10000):
        super(SinusoidalPosEmb, self).__init__()
        self.dim = dim
        self.max_positions = max_positions

    def call(self, x, training=True):
        x = tf.cast(x, tf.float32)
        half_dim = self.dim // 2
        emb = tf.exp(tf.range(half_dim, dtype=tf.float32) * -tf.math.log(self.max_positions) / (half_dim - 1))
        emb = x[:, None] * emb[None, :]
        emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)
        return emb

class Unet(tf.keras.Model):
    def __init__(self, dim=64, dim_mults=(1, 2, 4), channels=1):
        super(Unet, self).__init__()
        self.channels = channels
        init_dim = dim // 3 * 2
        self.init_conv = nn.Conv2D(filters=init_dim, kernel_size=7, strides=1, padding='SAME')

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        self.downs = []  # Downsampling layers
        self.ups = []    # Upsampling layers

        block_klass = partial(self._resnet_block)
        time_dim = dim * 4

        self.time_mlp = nn.Dense(units=time_dim)

        for ind, (dim_in, dim_out) in enumerate(zip(dims[:-1], dims[1:])):
            self.downs.append(self._create_downsample_block(dim_in, dim_out, time_dim, block_klass))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim)
        self.mid_attn = nn.Attention()
        self.mid_block2 = block_klass(mid_dim, mid_dim)

        for ind, (dim_in, dim_out) in enumerate(zip(reversed(dims[1:]), reversed(dims[:-1]))):
            self.ups.append(self._create_upsample_block(dim_in, dim_out, time_dim, block_klass))

        default_out_dim = channels * 2
        self.final_conv = nn.Conv2D(filters=default_out_dim, kernel_size=1, strides=1)

    def call(self, x, time, training=True):
        x = self.init_conv(x)
        t = self.time_mlp(time)

        h = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = tf.concat([x, h.pop()], axis=-1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        x = self.final_conv(x)
        return x

    def _create_downsample_block(self, dim_in, dim_out, time_dim, block_klass):
        return [
            block_klass(dim_in, dim_out),
            block_klass(dim_out, dim_out),
            nn.Attention(),
            nn.Conv2D(filters=dim_out, kernel_size=4, strides=2, padding='SAME')
        ]

    def _create_upsample_block(self, dim_in, dim_out, time_dim, block_klass):
        return [
            block_klass(dim_out * 2, dim_in),
            block_klass(dim_in, dim_in),
            nn.Attention(),
            nn.Conv2DTranspose(filters=dim_in, kernel_size=4, strides=2, padding='SAME')
        ]

    def _resnet_block(self, dim_in, dim_out):
        return nn.Sequential([
            nn.Conv2D(filters=dim_out, kernel_size=3, strides=1, padding='SAME'),
            nn.LayerNormalization(),
            nn.ReLU(),
            nn.Conv2D(filters=dim_out, kernel_size=3, strides=1, padding='SAME'),
            nn.LayerNormalization()
        ])
