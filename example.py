from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys

text = open('story.txt', 'r')
text = text.read()
print('corpus length:',len(text))

chars = sorted(list(set(text)))
print('total chars:',len(chars))
char_indices = dict((c,i) for i, c in enumerate(chars))
indices_char = dict((i,c) for i, c in enumerate(chars))

maxlen = 30
step = 3
sentences = []
next_chars = []

for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentences in enumerate(sentences):
    for t, char in enumerate(sentences):
        X[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1

#/////////////////////////////////////////////


TypedWord = 'summer'

print('Build model...')
model = Sequential()
model.add(LSTM(256, return_sequences=False, input_shape=(maxlen, len(chars))))
# model.add(LSTM(512, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

def sample(a, temperature=1.0):
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

newtext = open('forseed.txt', 'w')

for i in range(0,len(TypedWord)):
    Text = open('story.txt', 'r')
    Text = Text.read()
    for word in Text.split():
        if word[0] == TypedWord[i] or word[0] == TypedWord[i].upper:
            newtext.write(word)
            newtext.write(' ')
    if i != len(TypedWord)-1:
        newtext.write('\n')

newtext.close()

seedtext = open('forseed.txt','r')
seedlist = seedtext.read().split('\n')

for interation in range(1, 6):
    print('-' * 50)
    print('Iteration', interation)
    model.fit(X, y, batch_size=100, nb_epoch=20)
	
    for diversity in [0.2, 1.2]:
        print()
        print('----- diversity:', diversity)
        for idx in range(0,len(TypedWord)):
            seeds = seedlist[idx].split()
            for j in range (0, 10):
                sentence = random.choice(seeds)
                print('----- Generating with seed: "' + sentence + '"')
                if sentence[len(sentence)-1] == '.' or sentence[len(sentence)-1] == '?' or sentence[len(sentence)-1] == '!':
                    print (sentence)
                else:
                    generated = ''
                    generated += sentence
                    sys.stdout.write(generated)
                    for i in range(50):
                        x = np.zeros((1, maxlen, len(chars)))
                        for t, char in enumerate(sentence):
                            x[0, t, char_indices[char]] = 1.

                        preds = model.predict(x, verbose=0)[0]
                        next_index = sample(preds, diversity)
                        next_char = indices_char[next_index]

                        generated += next_char
                        sentence = sentence[1:] + next_char
                        sys.stdout.write(next_char)
                        sys.stdout.flush()
                    print()

"""
for interation in range(1, 10):
	print()
	print('-' * 50)
	print('Iteration', interation)
	model.fit(X, y, batch_size=20, nb_epoch=1)

	start_index = random.randint(0, len(text) - maxlen - 1)

	for diversity in [0.2, 0.5, 1.0, 1.2]:
		print()
		print('----- diversity:', diversity)

		generated = ''
		sentence = text[start_index: start_index + maxlen]
		generated += sentence
		print('----- Generating with seed: "' + sentence + '"')
		sys.stdout.write(generated)

		for i in range(400):
			x = np.zeros((1, maxlen, len(chars)))
			for t, char in enumerate(sentence):
				x[0, t, char_indices[char]] = 1.

			preds = model.predict(x, verbose=0)[0]
			next_index = sample(preds, diversity)
			next_char = indices_char[next_index]

			generated += next_char
			sentence = sentence[1:] + next_char
			sys.stdout.write(next_char)
			sys.stdout.flush()
		print()
"""