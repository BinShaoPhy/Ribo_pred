#https://keras.io/examples/nlp/text_classification_with_transformer/
import os
from tensorflow import Tensor
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
import keras_tuner as kt
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
import time
import sys

from keras.losses import mean_squared_error

from Ribo_pred_model import*

path = os.path.dirname(os.getcwd()) + "/RiboSTD_data/"

x_c = np.loadtxt(path + 'Cm_Mg_new_codon_xc.txt', delimiter="\t")
y_c = np.loadtxt(path + 'Cm_Mg_new_codon_yc.txt', delimiter="\t")

print(x_c.shape, y_c.shape)

x_c[:,0:40] = x_c[:,0:40]/100-5
x_c[:,40] = x_c[:,40]/100
y_c = y_c/100-5

num_val_samples = int(0.15 * x_c.shape[0])
num_train_samples = x_c.shape[0] - 2 * num_val_samples
indices = np.random.permutation(x_c.shape[0])


training_idx, test_idx = indices[:num_train_samples], indices[num_train_samples:num_train_samples + num_val_samples]
val_idx = indices[num_train_samples + num_val_samples:]

(x_train, y_train) = (x_c[training_idx], y_c[training_idx])
(x_val, y_val) = (x_c[val_idx], y_c[val_idx])
(x_test, y_test) = (x_c[test_idx], y_c[test_idx])

# restore np.load for future normal usage
# np.load = np_load_old
print(len(x_train), "Training sequences")
print(len(x_val), "Validation sequences")
print(len(x_test), "Testing sequences")

print(x_train[0])

inputs1 = layers.Input(shape=(40,))
maxlen = x_c.shape[1]
embedding_layer = TokenAndPositionEmbedding(40, vocab_size, embed_dim)
x1 = embedding_layer(inputs1)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
transformer_block2 = TransformerBlock(embed_dim, num_heads, ff_dim)
x1, outputs2 = transformer_block(x1, training = True)
x1 = layers.Reshape(target_shape = (40,8,1))(x1)
x1 = layers.Conv2D(kernel_size = kernal_size, filters = ff_num, padding = "same")(x1)
x1 = relu_bn(x1)
x1 = layers.Conv2D(kernel_size = kernal_size, filters = ff_num, padding = "same")(x1)
x1 = relu_bn(x1)
x1 = layers.Conv2D(kernel_size = kernal_size, filters = ff_num, padding = "same")(x1)
x1 = relu_bn(x1)
x1 = layers.Conv2D(kernel_size = kernal_size, filters = ff_num, padding = "same")(x1)
x1 = relu_bn(x1)
x1 = layers.Conv2D(kernel_size = kernal_size, filters = ff_num, padding = "same")(x1)
x1 = relu_bn(x1)
x1 = layers.Conv2D(kernel_size = kernal_size, filters = ff_num, padding = "same")(x1)
x1 = relu_bn(x1)
x1 = layers.Conv2D(kernel_size = kernal_size, filters = 32, padding = "same")(x1)
x1 = relu_bn(x1)
x1 = layers.GlobalAveragePooling2D()(x1)
x1 = layers.Dropout(0.2)(x1)

outputs1 = layers.Dense(1, activation="relu")(x1)

model = keras.Model(inputs=[inputs1], outputs=[outputs1])

opt = tf.keras.optimizers.Adam(learning_rate=0.002)
#model.compile(optimizer=opt, loss="mean_squared_error",loss_weights=[1., 0.0], metrics=["accuracy"])
model.compile(optimizer=opt, loss="mean_squared_error", metrics=["accuracy"])
print(model.summary())
history = model.fit(
    [x_train[:, 42:82]], y_train, batch_size=64, epochs=10, \
    validation_data=([x_val[:, 42:82]], y_val)
)

val_acc_per_epoch = history.history['val_loss']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

N_test = len(x_test)
plt.plot(y_test[0:N_test] , model.predict([x_test[0:N_test, 42:82]]),'.', markersize = 2)
# plt.plot(y_train[0:10000], model.predict([x_train[0:10000, 42:82],x_train[0:10000,0:42]]),'.', markersize = 2)
plt.savefig('x_train_codon_test' + '.png', dpi = 400, transparent=False)
correlation_matrix = np.corrcoef(model.predict([x_test[0:N_test, 42:82]]).reshape(1,N_test),\
                                 (y_test[0:N_test]).reshape(1,N_test))
print("model prediction %f" % (correlation_matrix[0][1]))
correlation_matrix = np.corrcoef(x_test[0:N_test,20].reshape(1,N_test), y_test[0:N_test].reshape(1,N_test))
print("original correlation %f" % (correlation_matrix[0][1]))

# np.save('my_history_larger_attn.npy',history.history)
model.save(path + 'deepNN'+".h5")


