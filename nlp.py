import numpy as np

from tqdm import tqdm
from keras.preprocessing.text import Tokenizer

def extract_glove_embeddings(text, glove, tokenizer, return_embeddings=True):
    glove_words = [datum.split()[0] for datum in glove]

    tokenizer.fit_on_texts(text)
    vocab_size = tokenizer.num_words

    common_words = [word for word in tokenizer.word_index.keys() if word in glove_words]
    
    tokenizer.word_index = {common_words[i]: i for i in range(len(common_words))}

    if not return_embeddings:
        return

    if tokenizer.num_words is None:
        vocab_size = len(tokenizer.word_index)
    
    word_idx_glove = {word: np.where(np.array(glove_words) == word)[0][0]for word in tqdm(tokenizer.word_index.keys())}

    glove_embeddings = np.zeros((vocab_size, 50))
    for i, glove_idx in enumerate(word_idx_glove.values()):
        if i == vocab_size:
            break
        glove_embeddings[i] = np.array([float(n) for n in glove[i].split()[1:]])

    return glove_embeddings