import tensorflow as tf
import tensorflow.keras.layers as nn
import tensorflow_addons as tfa
from tensorflow.keras import Sequential
from einops import rearrange
from functools import partial

class SinusoidalPosEmb(nn.Layer):
    def __init__(self, dim, max_positions=10000):
        super(SinusoidalPosEmb, self).__init__()
        self.dim = dim
        self.max_positions = max_positions

    def call(self, x, training=True):
        x = tf.cast(x, tf.float32)
        half_dim = self.dim // 2
        emb = tf.exp(tf.range(half_dim, dtype=tf.float32) * -math.log(self.max_positions) / (half_dim - 1))
        emb = x[:, None] * emb[None, :]
        emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)
        return emb

class Unet(Model):
    def __init__(self, dim=64, dim_mults=(1, 2, 4), channels=1, resnet_block_groups=8):
        super(Unet, self).__init__()
        self.channels = channels
        init_dim = dim // 3 * 2
        self.init_conv = nn.Conv2D(filters=init_dim, kernel_size=7, strides=1, padding='SAME')

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        self.downs = []  # Downsampling layers
        self.ups = []    # Upsampling layers

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)
        time_dim = dim * 4

        # Time embedding layers
        self.time_mlp = Sequential([
            SinusoidalPosEmb(dim),
            nn.Dense(units=time_dim),
            nn.GELU(),
            nn.Dense(units=time_dim)
        ])

        # Add downsample and upsample layers
        for ind, (dim_in, dim_out) in enumerate(zip(dims[:-1], dims[1:])):
            self.downs.append(self._create_downsample_block(dim_in, dim_out, time_dim, block_klass))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim)
        self.mid_attn = nn.Attention(mid_dim)
        self.mid_block2 = block_klass(mid_dim, mid_dim)

        for ind, (dim_in, dim_out) in enumerate(zip(reversed(dims[1:]), reversed(dims[:-1]))):
            self.ups.append(self._create_upsample_block(dim_in, dim_out, time_dim, block_klass))

        default_out_dim = channels * 2  # Output layer
        self.final_conv = Sequential([
            block_klass(dim * 2, dim),
            nn.Conv2D(filters=default_out_dim, kernel_size=1, strides=1)
        ])

    def call(self, x, time, training=True):
        x = self.init_conv(x)
        t = self.time_mlp(time)

        # Downsample
        h = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        # Middle
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # Upsample
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
            block_klass(dim_in, dim_out, time_emb_dim=time_dim),
            block_klass(dim_out, dim_out, time_emb_dim=time_dim),
            nn.Attention(dim_out),
            nn.Conv2D(filters=dim_out, kernel_size=4, strides=2, padding='SAME')
        ]

    def _create_upsample_block(self, dim_in, dim_out, time_dim, block_klass):
        return [
            block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim),
            block_klass(dim_in, dim_in, time_emb_dim=time_dim),
            nn.Attention(dim_in),
            nn.Conv2DTranspose(filters=dim_in, kernel_size=4, strides=2, padding='SAME')
        ]
