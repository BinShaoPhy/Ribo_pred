from tensorflow import Tensor
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

# Transformer block with multi-head attention
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, drop_rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)

        self.att = layers.MultiHeadAttention(num_heads=num_heads, 
                                             key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), 
             layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(drop_rate)
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim

    def call(self, inputs, training):
        attn_output, attn_scores = self.att(inputs, inputs, return_attention_scores = True)
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

# positional embedding for the input layer
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super(TokenAndPositionEmbedding, self).__init__(**kwargs)

        self.token_embed = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_embed = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
        
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=self.maxlen, delta=1)
        positions = self.pos_embed(positions)
        x = self.token_embed(x)
        return x + positions

    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "maxlen": self.maxlen,
            "embed_dim": self.embed_dim,
        })
        return config

# Transformer model with conv layers
class DeepTransModel():
    def __init__(
        self,
        input_dim = 40,
        vocab_size = 64,
        embed_dim = 8,
        num_heads = 1,
        ff_dim = 32,
        ff_num = 32,
        kernal_size = 3,
        layer_num = 6,
        learning_rate = 0.0001,
        batch_size = 64,
    ):
        # input layers
        inputs = layers.Input(shape=(input_dim, ))
        embedding_layer = TokenAndPositionEmbedding(input_dim, vocab_size, embed_dim)
        x1 = embedding_layer(inputs)
        transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
        x1, outputs2 = transformer_block(x1, training = True)
        x1 = layers.Reshape(target_shape = (input_dim ,embed_dim, 1))(x1)
        
        # Convolution layers
        for _ in range(layer_num):
            x1 = layers.Conv2D(kernel_size = kernal_size, filters = ff_num, padding = "same")(x1)
            x1 = relu_bn(x1)

        x1 = layers.GlobalAveragePooling2D()(x1)
        x1 = layers.Dropout(0.2)(x1)

        outputs = layers.Dense(1, activation="relu")(x1)
        model = keras.Model(inputs=[inputs], outputs=[outputs])
       
        self.model = model
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def train(self, X, y, n_epochs=100):
        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=3),
            keras.callbacks.History(),
        ]
        
        # cosine decay learning schedule
        starter_learning_rate = self.learning_rate * 5
        end_learning_rate = self.learning_rate
        decay_steps = n_epochs * X.shape[0] / self.batch_size
        
        learning_rate_fn = tf.keras.optimizers.schedules.CosineDecay(
            starter_learning_rate,
            decay_steps,
            alpha = 0.0)
        
        opt = tf.keras.optimizers.Adam(learning_rate = learning_rate_fn)

        self.model.compile(optimizer=opt, loss="mean_squared_error", metrics=["accuracy"])
        
        # model fitting
        history = self.model.fit(
            x=X,
            y=y,
            epochs=n_epochs,
            shuffle=True,
            validation_split=0.2,
            verbose = 1,
            batch_size=self.batch_size,
            callbacks=callbacks,
        )
        
        return history
    
    def save(self, h5file):
        self.model.save(h5file)

    def predict(self, X, batch_size):
        return self.model.predict(X, batch_size)
    
    def evaluate(self, X, y, batch_size, verbose):
        return self.model.evaluate(
            X, y, batch_size, verbose
        ) 
    
    def to_json(self):
        return self.model.to_json()


def relu_bn(inputs: Tensor) -> Tensor:
    relu = layers.ReLU()(inputs)
    bn = layers.BatchNormalization()(relu)
    return bn