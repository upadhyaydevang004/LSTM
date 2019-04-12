import os
import numpy as np
from keras.models import Sequential
from keras import losses
from keras import metrics
from keras import backend as K
from keras import optimizers
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import pickle

seqlength = 32
batch_size = 128
epochs = 10
thresh_unk = 100
drop_prob = 0.4

def oneHot(index, len):
	arr = np.zeros(len)
	arr[index] = 1
	return arr

# load doc into memory
def load_doc(filename):
	file = open(filename, 'r')
	text = file.read()
	file.close()
	return text


path = ''
train_file = path + 'ptb.char.train.txt'     
val_file = path + 'ptb.char.valid.txt'

train_raw_text = load_doc(train_file)
val_raw_text = load_doc(val_file)

X_train_char = train_raw_text.split(' ')
X_CV_char = val_raw_text.split(' ')



# Create vocabulary
vocab = sorted(list(set(X_train_char)))

unk = []


for i in range(len(vocab)):
    if(X_train_char.count(vocab[i]) <= thresh_unk):
        unk.append(vocab[i])

print('UNK = ', unk)

for i in range(len(unk)):
    vocab = [temp for temp in vocab if temp != unk[i]]
    X_train_char = [temp for temp in X_train_char if temp != unk[i]]
    X_CV_char = [temp for temp in X_CV_char if temp != unk[i]]

vocab_size = len(vocab)

X_train_char = X_train_char[:-2]
X_CV_char = X_CV_char[:-2]
Y_train_char = []
Y_CV_char = []

for i in range(0,len(X_train_char)-2):
	Y_train_char.append(X_train_char[i+1])
for i in range(0,len(X_CV_char)-2):
    Y_CV_char.append(X_CV_char[i+1])

X_train_char = X_train_char[:-2]
X_CV_char = X_CV_char[:-2]

print(len(X_train_char))
print(len(Y_train_char))
print(len(X_CV_char))
print(len(Y_CV_char))

char2int = dict((c, i) for i, c in enumerate(vocab))
int2char = dict((i, c) for i, c in enumerate(vocab))


X_train_oneHot = []
Y_train_oneHot = []
X_CV_oneHot = []
Y_CV_oneHot = []

len_train = 500000             
len_CV = 50000                 

print('Converting char to one hot vectors')
for i in range(len_train):
	X_train_oneHot.append( oneHot(char2int[X_train_char[i]], vocab_size) )
	Y_train_oneHot.append( oneHot(char2int[Y_train_char[i]], vocab_size) )

for i in range(len_CV):
	X_CV_oneHot.append( oneHot(char2int[X_CV_char[i]], vocab_size) )
	Y_CV_oneHot.append( oneHot(char2int[Y_CV_char[i]], vocab_size) )




X_train = []
Y_train = []
X_CV = []
Y_CV = []


print('creating sequences for input')
for i in range(0, len(X_train_oneHot)-seqlength):
    X_train.append(X_train_oneHot[i:i+seqlength])
    Y_train.append(Y_train_oneHot[i+seqlength])

for i in range(0, len(X_CV_oneHot)-seqlength):
    X_CV.append(X_CV_oneHot[i:i+seqlength])
    Y_CV.append(Y_CV_oneHot[i+seqlength])


print('Converting to numpy array')
X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_CV = np.array(X_CV)
Y_CV = np.array(Y_CV)


print('Shufling data..')
X_train, _, Y_train, _ = train_test_split(X_train, Y_train, test_size=0, random_state=2)

print(X_train.shape)
print(Y_train.shape)
print(X_CV.shape)
print(Y_CV.shape)


print('Building model')
model = Sequential()
model.add(LSTM(500, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True ))
model.add(Dropout(drop_prob))
model.add(LSTM(500))
model.add(Dropout(drop_prob))
model.add(Dense(vocab_size, activation='softmax'))

print(model.summary())
print('\nDropout : ', drop_prob, 
      '\nBatch Size : ', batch_size,
      '\nEpochs : ', epochs,
      '\nSeq Length : ', seqlength)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

checkpoint = ModelCheckpoint('weights.hdf5', monitor='val_acc', save_best_only=True, verbose=1, mode='max')
callbacks_list = [checkpoint]

hist_obj = model.fit(X_train, Y_train, validation_data=[X_CV, Y_CV], epochs=epochs, batch_size=batch_size, callbacks=callbacks_list, verbose=1)

train_loss = hist_obj.history['loss']
val_loss = hist_obj.history['val_loss']
train_acc = hist_obj.history['acc']
val_acc = hist_obj.history['val_acc']


with open('train_loss.pkl', 'wb') as f:
    pickle.dump(train_loss, f)   

with open('val_loss.pkl', 'wb') as f:
    pickle.dump(val_loss, f)

with open('train_acc.pkl', 'wb') as f:
    pickle.dump(train_acc, f)

with open('val_acc.pkl', 'wb') as f:
    pickle.dump(val_acc, f)

print('Metrics saved')
