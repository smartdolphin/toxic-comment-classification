import argparse
from keras import backend as K
from keras import callbacks
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd
import data_util
from model import classifier_model
import warnings
warnings.filterwarnings('ignore')


def schedule(ind):
    a = [0.001, 0.0005, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001]
    return a[ind]


def main():
    args = argparse.ArgumentParser()
    args.add_argument('--max_features', type=int, default=100000)
    args.add_argument('--max_length', type=int, default=200)
    args.add_argument('--embed_size', type=int, default=300)
    args.add_argument('--units', type=int, default=80)
    args.add_argument('--dropout_rate', type=float, default=0.2)
    args.add_argument('--output_num', type=int, default=6)
    args.add_argument('--filter_num', type=int, default=64)
    args.add_argument('--filter_size', type=int, default=3)
    args.add_argument('--batch_size', type=int, default=160)
    args.add_argument('--epochs', type=int, default=10)
    args.add_argument('--train_size', type=float, default=0.95)
    args.add_argument('--split_random_state', type=int, default=233)
    args.add_argument('--train_file_path', type=str, default='data/train.csv')
    args.add_argument('--test_file_path', type=str, default='data/test.csv')
    args.add_argument('--output_file_path', type=str, default='submission_result.csv')
    config = args.parse_args()

    session_conf = tf.ConfigProto(intra_op_parallelism_threads=4, inter_op_parallelism_threads=4)
    K.set_session(tf.Session(graph=tf.get_default_graph(), config=session_conf))

    x_train, x_test, y_train, embedding_matrix = data_util.read_data(config.train_file_path,
                                                                     config.test_file_path,
                                                                     config.max_features,
                                                                     config.max_length,
                                                                     config.embed_size)

    model = classifier_model(config.units,
                             config.max_length,
                             config.max_features,
                             config.output_num,
                             config.embed_size,
                             config.dropout_rate,
                             embedding_matrix,
                             config.filter_num,
                             config.filter_size)

    x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                      y_train,
                                                      train_size=config.train_size,
                                                      random_state=config.split_random_state)

    lr = callbacks.LearningRateScheduler(schedule)
    ra_val = data_util.RocAucEvaluation(validation_data=(x_val, y_val), interval=1)
    early_stopping = EarlyStopping(patience=1, verbose=2)

    hist = model.fit(x_train,
                     y_train,
                     batch_size=config.batch_size,
                     epochs=config.epochs,
                     validation_data=(x_val, y_val),
                     callbacks=[lr, ra_val, early_stopping],
                     verbose=1)
    model.save('toxic_comment_{0}.h5'.format(config.epochs))
    data_util.visual(hist, 'toxic_comment_{0}'.format(config.epochs))
    print('Training complete')

    y_pred = model.predict(x_test, batch_size=config.batch_size, verbose=0)
    submission = pd.read_csv('data/sample_submission.csv')
    submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_pred
    submission.to_csv(config.output_file_path, index=False)
    print('Generated submission file')


if __name__ == '__main__':
    main()
