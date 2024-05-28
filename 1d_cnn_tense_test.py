# Description: 1D cnn-based models for tense detection.

import json
import os

from keras.layers import Embedding, Conv1D, Dense,Activation
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
import numpy as np
from keras.models import load_model
from keras.models import Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt

EMBEDDING_DIM = 300
# model configruation
MAX_SEQUENCE_LENGTH = 320
BATCH_SIZE = 256
NB_EPOCHS = 2
sent_list = 411319

def get_verb_info(verb_lines):
    verbs_info = dict()
    for verb_ln in verb_lines:
        verb_data = verb_ln.rstrip('\t').split('\t')
        verb_positions = verb_data[0].split(" ")
        verb_tense = verb_data[2]
        if verb_tense in {'cond other', 'pres other'}:  # we don't want these rare & strange tenses
            verb_tense = 'other'
        for verb_position in verb_positions:
            verbs_info[int(verb_position)] = verb_tense
    return verbs_info


def get_tagged_tokens(sentence, verbs_information):
    tagged_tokens = []
    for token_index, token in enumerate(sentence.split(" "), 1):
        tag = verbs_information[token_index] if token_index in verbs_information else 'O'

        tagged_tokens.append((token, tag))
    #print(tagged_tokens[0])
    return tagged_tokens


# prepare the token_list from the tags.txt or dataset.
def prepare_tag_list(tg_path, ds_path):
    tags = []
    # if exists tg_path, read directly, no need to read from dataset.
    if os.path.exists(tg_path):
        with open(tg_path, "r") as tg_file:
            tags = json.loads(tg_file.read())
    elif os.path.exists(ds_path):
        with open(ds_path, 'r') as ds_file:
            ds_tags_l = []
            # Remove any white space
            all = ds_file.read().rstrip("\n")
            fields = all.split("\n\n")
            # deal with every field in the dataset
            for field in fields:
                lines = field.split("\n")
                verbs_info_dict = get_verb_info(lines[2:])
                ds_tags_l.extend(verbs_info_dict.values())
            tags = sorted(list(set(ds_tags_l)), key=ds_tags_l.index)
        # save tags
        with open(tg_path, "w") as f:
            f.write(json.dumps(tags))
    else:
        assert (len(tags) == 0)
    tags.append('O')  # a default tag for all non-verbs
    return tags


# generate one sentence with split-ed words and corresponding tags.
def tagged_sentence_generator(ds_path, start_sent_nb=0, end_sent_nb_excl=None):
    with open(ds_path, 'r') as f:
        all = f.read().rstrip("\n")
        fields = all.split("\n\n")
        for field in fields[start_sent_nb: end_sent_nb_excl]:
            lines = field.split("\n")
            english_sent = lines[0]
            verbs_info_dict = get_verb_info(lines[2:])
            tagged_sentence = get_tagged_tokens(english_sent, verbs_info_dict)
            #print(tagged_sentence)
            yield tagged_sentence


# prepare the token list from the dictionary.txt or dataset.
def prepare_token_list(dict_path, ds_path=None):
    dictionary = dict()
    if os.path.exists(dict_path):
        with open(dict_path, "r") as f:
            dictionary = json.loads(f.read())
    elif os.path.exists(ds_path):
        words = set([])
        sent_generator = tagged_sentence_generator(ds_path=ds_path)

        for sent in sent_generator:
            print(sent)
        [[words.add(token) for token, _ in sent] for sent in sent_generator]

        print("Vocabulary loading from the dictionary, size: {}".format(len(words)))

        for i, word in enumerate(words):
            dictionary[word] = i
        with open(dict_path, "w") as f:
            f.write(json.dumps(dictionary))
    else:
        assert (len(dictionary) == 0)
    return dictionary
    # the vocabulary


def load_ds_sentences(ds_path):
    sents = []
    with open(ds_path, 'r') as f:
        all = f.read().rstrip("\n")
        fields = all.split("\n\n")
        for field in fields:
            lines = field.split("\n")
            english_sent = lines[0]
            verbs_info_dict = get_verb_info(lines[2:])
            sents.append(get_tagged_tokens(english_sent, verbs_info_dict))
    return sents


def prepare_word_embeddings(we_path):
    # let's create a dictionary of embeddings from each word embedding vector in the pre-trained GloVe embeddings file
    embeddings_index = {}
    f = open(we_path, encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print(values)
    #print(embeddings_index)
    print('Found %s word vectors.' % len(embeddings_index))  # 400000

    # let's try to extract the GloVe embeddings for each word from our dataset vocabulary
    EMBEDDING_DIM = 300
    embedding_matrix = np.zeros((len(token_dict) + 1, EMBEDDING_DIM))
    for word, i in token_dict.items():
        embedding_vector = embeddings_index.get(word)
        # print(embedding_vector)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i, :] = embedding_vector
    # print()
    return embedding_matrix

# load data
def prepare_batch(x_batch, y_batch, nb_labels, tok_dict, tag2label):
    # pad sequences with zeros to make them same length: we need it for vectorized computations
    x_batch = pad_sequences(x_batch, maxlen=MAX_SEQUENCE_LENGTH, padding='post', value=tok_dict[''])
    # convert labels to categorical one-hot vectors
    y_batch = pad_sequences(y_batch, maxlen=MAX_SEQUENCE_LENGTH, padding='post', value=tag2label['O'])
    y_batch = to_categorical(y_batch, nb_labels)  # multiclass data
    return np.array(x_batch), np.array(y_batch, dtype="float64")


# only prepare x_batch.
def prepare_test_batch(x_batch, tok_dict):
    # pad sequences with zeros to make them same length: we need it for vectorized computations
    x_batch = pad_sequences(x_batch, maxlen=MAX_SEQUENCE_LENGTH, padding='post', value=tok_dict[''])
    return np.array(x_batch)


def batch_generator(
        sent_list,
        max_sent_token_len,
        start_idx, end_idx_excl,
        tok_dict, tag2label, nb_labels
):
    """infinitely yield batches of sequences and labels"""
    x_batch, y_batch = [], []
    while True:
        data_list = sent_list[start_idx: end_idx_excl]
        print(data_list)
        for sentence in data_list:
            toks, tags = zip(*sentence[:max_sent_token_len])  # !!! we cut off too long sentences !!!
            # convert sentence into sequences of word indexes
            sequence = [tok_dict[tok] for tok in toks]  # TODO: we might want to try lower-cased tokens
            labels = [tag2label[tag] for tag in tags]
            if len(x_batch) == BATCH_SIZE:
                yield prepare_batch(x_batch, y_batch, nb_labels, tok_dict, tag2label)
                x_batch, y_batch = [], []
            x_batch.append(sequence)
            y_batch.append(labels)
        # print(y_batch[2])
        if len(x_batch) != 0:
            yield prepare_batch(x_batch, y_batch, nb_labels, tok_dict, tag2label)


def print_predictions(x_test, y_pred, idx2word, lab2tag):
    """
    print the results of our model's predictions after converting them back to tokens and tags
    """
    pad_symbol = ''
    for seq, preds in zip(x_test, y_pred):
        sentence = []
        pad_removed = False
        for i in range(len(seq) - 1, -1, -1):
            word_id, pred = seq[i], preds[i]
            # lab2tag[]
            word, tag = idx2word[word_id], lab2tag[np.argmax(pred)]
            # print(np.argmax(pred))
            if word == pad_symbol and not pad_removed:
                continue
            else:
                pad_removed = True
                sentence.append((word, tag))
        print(list(reversed(sentence)))
        print('\n')


if __name__ == "__main__":
    ds_path = "./CorpusAnnotatedTenseVoice.txt"
    tg_path = "./tags.txt"
    dict_path = "./dictionary.txt"
    we_path = "./glove.6B.300d.txt"

    # prepare tags
    tag_list = prepare_tag_list(tg_path, ds_path)
    # print(tag_list)
    tag2lab = {}
    lab2tag = {}
    for i, tag in enumerate(tag_list):
        tag2lab[tag] = i
        lab2tag[i] = tag
    # print("labtag", lab2tag)
    # print("tag2lab", tag2lab)

    # prepare token dictionary.
    #print(token_dict)
    token_dict = prepare_token_list(dict_path, ds_path)

    #print("index of `hello`: {}".format(token_dict["hello"]))

    # a mapping for indexes back into words
    idx2word = {}
    for word, i in token_dict.items():
        idx2word[i] = word
    # prepare the dataset
    sent_list = load_ds_sentences(ds_path)
    ds_size = len(sent_list)
    # print(ds_size) 411319
    max_sent_len = 0  # max tokens in one sentences.
    for sent in sent_list:
        max_sent_len = max(max_sent_len, len(sent))
    print("Max. sentence length: {} tokens".format(max_sent_len))

    # our dataset will be split into a training part and a validation part,
    # where we measure our model's performance during training
    # we will further keep a testing part to evaluate predictions

    # embedding matrix
    df = prepare_word_embeddings(we_path)

    TEST_SPLIT = .025
    nb_test_samples = int(TEST_SPLIT * ds_size)  # 10282

    train_generator = batch_generator(sent_list, max_sent_len,
                                      0, ds_size - 2 * nb_test_samples, token_dict, tag2lab, len(tag_list)
                                      )
    # print(train_generator)
    # print("run train_ generator")
    val_generator = batch_generator(sent_list, max_sent_len,
                                    ds_size - 2 * nb_test_samples, ds_size - nb_test_samples, token_dict, tag2lab,
                                    len(tag_list)
                                    )
    test_generator = batch_generator(sent_list, max_sent_len,
                                     ds_size - nb_test_samples, ds_size, token_dict, tag2lab, len(tag_list)
                                     )
    steps_per_epoch = np.ceil((ds_size - nb_test_samples)/BATCH_SIZE)
    validation_steps = np.ceil(nb_test_samples/BATCH_SIZE)
    # TODO:train and evaluate model here
    output_dim = 10
    # load pre-trained
    embedding_layer = Embedding(len(token_dict) + 1,
                                input_length=MAX_SEQUENCE_LENGTH, weights=[df],
                                trainable=False)
    model = Sequential()
    #first layer
    model.add(embedding_layer)
    model.add(Masking(mask_value=0.0))
    # layer 2
    # Recurrent layer
   # model.add(LSTM())
    model.add(Dense(300, activation='relu'))
    # Dropout for regularization
    model.add(Dropout(0.2))
    # layer3
    model.add(Dense(150, activation='relu'))
    model.add(Dropout(0.2))
    #ouptut_layer
    model.add(Dense(units=20, activation='softmax'))
    print(model.summary())
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # fit network
    model.fit_generator(train_generator, epochs=NB_EPOCHS, steps_per_epoch=steps_per_epoch, verbose=1,
                        validation_data=val_generator, validation_steps=validation_steps)
    # evaluate the model
    loss, accuracy = model.history()
