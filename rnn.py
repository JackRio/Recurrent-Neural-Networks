import nltk
import numpy as np
from nltk.corpus import PlaintextCorpusReader

vocabulary_size = 8000
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

reader = PlaintextCorpusReader('./data/', 'reddit-comments-2015-08.csv', encoding='utf-8')
sentences = reader.sents()
words = reader.words()
tokenized_sentences = [[sentence_start_token] + sent + [sentence_end_token] for sent in sentences]

word_freq = nltk.FreqDist(words)
word_freq.plot(30)
print("Unique words", len(word_freq))

vocab = word_freq.most_common(vocabulary_size - 1)
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)
word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

print("Least freq word in vocab is:", f'"{vocab[-1][0]}"', "and it appeared", vocab[-1][1], "times")

for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = ([word if word in word_to_index else unknown_token for word in sent])

X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

