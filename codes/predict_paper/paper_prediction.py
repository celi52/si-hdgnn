"""
Author: Xovee Xu
"""
import sys
import os
BASE_PATH = os.path.abspath(os.path.join(os.getcwd()))
sys.path.append(BASE_PATH)
import pickle
import numpy as np
import tensorflow as tf
import config


with open(config.p_x, 'rb') as f:
    x = pickle.load(f)

with open(config.p_x_authors, 'rb') as f:
    x_authors = pickle.load(f)

with open(config.p_y, 'rb') as f:
    y = pickle.load(f)

# params
b_size = 16
learning_rate = 5e-4
verbose = 2
epochs = 1000
author_emb_dim = 128
paper_emb_dim = 256
rnn_units = 128
mlp_units = 64
max_seq = config.seq_length
max_authors = config.author_length
patience = 10
dropout = 0.1
weight_decay = 5e-4

print('Max # authors:', max_authors)
print('Max # sequences:', max_seq)
print('# authors:', len(x))

for temp in x_authors:
    for tem in temp:
        while len(tem) < max_authors:
            tem.append(np.zeros(author_emb_dim))
    while len(temp) < max_seq:
        temp.append(np.zeros(shape=(max_authors, author_emb_dim)))

x_authors = np.array(x_authors)

for temp in x:
    while len(temp) < max_seq:
        temp.append(np.zeros(temp[0].shape))

x = np.array(x)
y = np.array(y)

print('x shape', x.shape)
print('x authors shape:', x_authors.shape)
print('y shape:', y.shape)

train = x[:int(len(x)*.5)]
val = x[int(len(x)*.5):int(len(x)*.75)]
test = x[int(len(x)*.75):]

train_authors, train_y = x_authors[:int(len(x_authors)*.5)], y[:int(len(x_authors)*.5)]
val_authors, val_y = x_authors[int(len(x_authors)*.5):int(len(x_authors)*.75)], \
              y[int(len(x_authors)*.5):int(len(x_authors)*.75)]
test_authors, test_y = x_authors[int(len(x_authors)*.75):], y[int(len(x_authors)*.75):]

# del x_authors, x, y

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_msle', patience=patience,
                                              verbose=1, restore_best_weights=True)

# model
input = tf.keras.layers.Input(shape=(max_seq, paper_emb_dim))
input_authors = tf.keras.layers.Input(shape=(max_seq, max_authors, author_emb_dim))

masked_input = tf.keras.layers.Masking(mask_value=0., input_shape=(max_seq, rnn_units))(input)
masked_input_authors = tf.keras.layers.Masking(mask_value=0., input_shape=(max_seq, max_authors, author_emb_dim))(input_authors)

rnn_authors = tf.keras.layers.TimeDistributed(
    tf.keras.layers.GRU(rnn_units, dropout=dropout), input_shape=(max_seq, max_authors, author_emb_dim),
)(masked_input_authors)

con = tf.keras.layers.concatenate([masked_input, rnn_authors])

rnn_1 = tf.keras.layers.Bidirectional(
    tf.keras.layers.GRU(rnn_units, return_sequences=True, dropout=dropout)
)(con)

rnn_2 = tf.keras.layers.Bidirectional(
    tf.keras.layers.GRU(rnn_units//2, dropout=dropout)
)(rnn_1)


mlp_1 = tf.keras.layers.Dense(mlp_units, activation='relu')(rnn_2)
mlp_2 = tf.keras.layers.Dense(mlp_units//2, activation='relu')(mlp_1)
output = tf.keras.layers.Dense(1)(mlp_2)


xovee = tf.keras.Model(inputs=[input, input_authors], outputs=output)

adam = tf.keras.optimizers.RMSprop(learning_rate=learning_rate, decay=weight_decay)

xovee.compile(loss='msle', optimizer=adam, metrics=['msle'])

xovee.summary()

train_history = xovee.fit(x=(train, train_authors), y=train_y,
                          validation_data=((val, val_authors), val_y),
                          epochs=epochs,
                          verbose=verbose,
                          callbacks=[early_stop]
                          )

result_metric = xovee.evaluate(x=(test, test_authors), y=test_y,  verbose=1)
pred = xovee.predict(x=(test, test_authors))
