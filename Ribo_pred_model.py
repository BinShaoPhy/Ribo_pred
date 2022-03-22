from tensorflow import Tensor
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

# transformer variables
ff_num = 32 #32
kernal_size = 3
vocab_size = 64
embed_dim = 8  # Embedding size for each token
num_heads = 1  # 2 Number of attention heads
ff_dim = 32  # 32 Hidden layer size in feed forward network inside transformer

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim

        self.att = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(self.ff_dim, activation="relu"), layers.Dense(self.embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        # attn_output = self.att(inputs, inputs)
        # attn_output = self.dropout1(attn_output, training=training)
        # out1 = self.layernorm1(inputs + attn_output)
        # ffn_output = self.ffn(out1)
        # ffn_output = self.dropout2(ffn_output, training=training)
        # return self.layernorm2(out1 + ffn_output)
        attn_output, attn_scores = self.att(inputs, inputs, return_attention_scores =  True)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        return ffn_output, attn_scores

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
        })
        return config

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super(TokenAndPositionEmbedding, self).__init__(**kwargs)
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_emb = layers.Embedding(input_dim=self.vocab_size, output_dim=self.embed_dim)
        self.pos_emb = layers.Embedding(input_dim=self.maxlen, output_dim=self.embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=self.maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "maxlen": self.maxlen,
            "embed_dim": self.embed_dim,
        })
        return config

class STDmodel(keras.Model):
    def __int__(self):
        super().__init__()
        self.ff_num = 32  # 32
        self.kernal_size = 3
        self.vocab_size = 64
        self.embed_dim = 8  # Embedding size for each token
        self.num_heads = 2  # Number of attention heads
        self.ff_dim = 32  # Hidden layer size in feed forward network in transformer
        self.conv1 = layers.Conv2D(kernel_size=self.kernal_size, filters=self.ff_num, padding="same")
        self.transformer = TransformerBlock(self.embed_dim, self.num_heads, self.ff_dim)
        self.embed = TokenAndPositionEmbedding(40, self.vocab_size, self.embed_dim)
        self.reshape = layers.Reshape(target_shape=(40, 8, 1))
        self.pool = layers.GlobalAveragePooling2D()
        self.dropout = layers.Dropout(0.2)
        self.dense1 = layers.Dense(1, activation="relu")


    def call(self, inputs, training):
        inp1, inp2 = inputs
        # x1 = self.embed(inp1)
        # x1, _ = self.transformer(x1, training = training)
        # x1 = self.reshape(x1)
        # x1 = self.conv1(x1)
        # x1 = self.pool(x1)
        # x1 = self.dropout(x1)
        x1 = self.dense1(inp1)
        return x1


def OnePotEncoding(x):
    x_encode = np.zeros((len(x),40,64,1))
    for i in range(len(x)):
        for j in range(40):
            x_encode[i,j, int(x[i,j]), 0] = 1
    return x_encode

#another self-defined transformer:
def relu_bn(inputs: Tensor) -> Tensor:
    relu = layers.ReLU()(inputs)
    bn = layers.BatchNormalization()(relu)
    return bn