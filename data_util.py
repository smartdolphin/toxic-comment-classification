import os
import pandas as pd
from keras.preprocessing import text, sequence
from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score
import gc
import numpy as np
import re

# https://fasttext.cc/docs/en/english-vectors.html
EMBEDDING_FILE = 'data/crawl-300d-2M.vec'


def read_data(train_file_path, test_file_path, max_features, max_length, embed_size):
    assert os.path.isfile(train_file_path) and os.path.isfile(test_file_path)

    train = pd.read_csv(train_file_path)
    test = pd.read_csv(test_file_path)

    # pre-processing start
    repl = {
        "&lt;3": " good ",
        ":d": " good ",
        ":dd": " good ",
        ":p": " good ",
        "8)": " good ",
        ":-)": " good ",
        ":)": " good ",
        ";)": " good ",
        "(-:": " good ",
        "(:": " good ",
        "yay!": " good ",
        "yay": " good ",
        "yaay": " good ",
        "yaaay": " good ",
        "yaaaay": " good ",
        "yaaaaay": " good ",
        ":/": " bad ",
        ":&gt;": " sad ",
        ":')": " sad ",
        ":-(": " bad ",
        ":(": " bad ",
        ":s": " bad ",
        ":-s": " bad ",
        "&lt;3": " heart ",
        ":d": " smile ",
        ":p": " smile ",
        ":dd": " smile ",
        "8)": " smile ",
        ":-)": " smile ",
        ":)": " smile ",
        ";)": " smile ",
        "(-:": " smile ",
        "(:": " smile ",
        ":/": " worry ",
        ":&gt;": " angry ",
        ":')": " sad ",
        ":-(": " sad ",
        ":(": " sad ",
        ":s": " sad ",
        ":-s": " sad ",
        r"\br\b": "are",
        r"\bu\b": "you",
        r"\bhaha\b": "ha",
        r"\bhahaha\b": "ha",
        r"\bdon't\b": "do not",
        r"\bdoesn't\b": "does not",
        r"\bdidn't\b": "did not",
        r"\bhasn't\b": "has not",
        r"\bhaven't\b": "have not",
        r"\bhadn't\b": "had not",
        r"\bwon't\b": "will not",
        r"\bwouldn't\b": "would not",
        r"\bcan't\b": "can not",
        r"\bcannot\b": "can not",
        r"\bi'm\b": "i am",
        "m": "am",
        "r": "are",
        "u": "you",
        "haha": "ha",
        "hahaha": "ha",
        "don't": "do not",
        "doesn't": "does not",
        "didn't": "did not",
        "hasn't": "has not",
        "haven't": "have not",
        "hadn't": "had not",
        "won't": "will not",
        "wouldn't": "would not",
        "can't": "can not",
        "cannot": "can not",
        "i'm": "i am",
        "m": "am",
        "i'll": "i will",
        "its": "it is",
        "it's": "it is",
        "'s": " is",
        "that's": "that is",
        "weren't": "were not",
        "<3": "love",
    }

    keys = [i for i in repl.keys()]

    new_train_data = []
    new_test_data = []
    ltr = train["comment_text"].tolist()
    lte = test["comment_text"].tolist()
    for i in ltr:
        arr = str(i).split()
        xx = ""
        for j in arr:
            j = str(j).lower()
            if j[:4] == 'http' or j[:3] == 'www':
                continue
            if j in keys:
                # print("inn")
                j = repl[j]
            xx += j + " "
        new_train_data.append(xx)
    for i in lte:
        arr = str(i).split()
        xx = ""
        for j in arr:
            j = str(j).lower()
            if j[:4] == 'http' or j[:3] == 'www':
                continue
            if j in keys:
                j = repl[j]
            xx += j + " "
        new_test_data.append(xx)
    train["new_comment_text"] = new_train_data
    test["new_comment_text"] = new_test_data
    print("crap removed")

    trate = train["new_comment_text"].tolist()
    tete = test["new_comment_text"].tolist()
    for i, c in enumerate(trate):
        trate[i] = re.sub('[^a-zA-Z ?!]+', '', str(trate[i]).lower())
    for i, c in enumerate(tete):
        tete[i] = re.sub('[^a-zA-Z ?!]+', '', tete[i])
    train["comment_text"] = trate
    test["comment_text"] = tete
    print('only alphabets')

    train_comment = train["comment_text"].fillna("fillna").values
    test_comment = test["comment_text"].fillna("fillna").values
    y_train = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values

    tokenizer = text.Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(train_comment) + list(test_comment))

    train_cmt_seq = tokenizer.texts_to_sequences(train_comment)
    test_cmt_seq = tokenizer.texts_to_sequences(test_comment)

    x_train = sequence.pad_sequences(train_cmt_seq, maxlen=max_length)
    x_test = sequence.pad_sequences(test_cmt_seq, maxlen=max_length)

    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(
        get_coefs(*o.rstrip().rsplit(' '))
        for i, o in enumerate(open(EMBEDDING_FILE, encoding="utf8")) if i != 0)

    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()

    del all_embs, train_comment, test_comment, train_cmt_seq, test_cmt_seq, train, test
    gc.collect()

    word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    print('pre-processing done')
    return x_train, x_test, y_train, embedding_matrix


def visual(hist, file_name):
    import matplotlib.pyplot as plt
    flg, loss_ax = plt.subplots()

    acc_ax = loss_ax.twinx()

    loss_ax.plot(hist.history['loss'], 'y', label='train loss')
    loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
    loss_ax.set_ylim([0.0, 3.0])

    acc_ax.plot(hist.history['acc'], 'b', label='train acc')
    acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')
    acc_ax.set_ylim([0.0, 1.0])

    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    acc_ax.set_ylabel('accuracy')

    loss_ax.legend(loc='upper left')
    acc_ax.legend(loc='lower left')

    plt.savefig('{0}.png'.format(file_name))


class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, batch_size=160, verbose=1)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch + 1, score))