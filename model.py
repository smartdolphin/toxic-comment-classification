from keras import Model
from keras.layers import Input, Dense, Embedding, Conv1D
from keras.layers import GRU, Bidirectional
from keras.layers import SpatialDropout1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import concatenate
from keras import optimizers


def classifier_model(units,
                     max_length,
                     max_features,
                     output_num,
                     embedding_size,
                     dropout_rate,
                     embedding_matrix,
                     filter_num,
                     filter_size,
                     non_static=False,
                     zero_masking=False):
    inp = Input(shape=(max_length,))
    emb = Embedding(max_features, embedding_size, weights=[embedding_matrix],
                    trainable=non_static, mask_zero=zero_masking)(inp)
    x = SpatialDropout1D(dropout_rate)(emb)
    x = Bidirectional(GRU(units, return_sequences=True))(x)
    x = Conv1D(filter_num, kernel_size=filter_size, padding="valid", kernel_initializer="he_uniform")(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    x = concatenate([avg_pool, max_pool])
    x = Dense(output_num, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.Adam(),
                  metrics=['accuracy'])
    model.summary()
    return model
