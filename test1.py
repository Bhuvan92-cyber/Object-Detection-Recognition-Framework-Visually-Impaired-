import os
import cv2
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle
from keras.applications.inception_v3 import InceptionV3

# Non-Binary Image Classification using Convolution Neural Networks
'''
class_labels = ['fifty', 'fivehundred', 'hundred', 'ten', 'thousand', 'twenty']

path = 'Dataset'

labels = []
X = []
Y = []

def getID(name):
    index = 0
    for i in range(len(labels)):
        if labels[i] == name:
            index = i
            break
    return index        
    

for root, dirs, directory in os.walk(path):
    for j in range(len(directory)):
        name = os.path.basename(root)
        if name not in labels:
            labels.append(name)
print(labels)



for root, dirs, directory in os.walk(path):
    for j in range(len(directory)):
        name = os.path.basename(root)
        print(name+" "+root+"/"+directory[j])
        if 'Thumbs.db' not in directory[j]:
            img = cv2.imread(root+"/"+directory[j])
            img = cv2.resize(img, (32,32))
            im2arr = np.array(img)
            im2arr = im2arr.reshape(32,32,3)
            X.append(im2arr)
            Y.append(getID(name))
        
X = np.asarray(X)
Y = np.asarray(Y)
print(Y)

np.save('model/X.txt',X)
np.save('model/Y.txt',Y)
'''
X = np.load('model/X.txt.npy')
Y = np.load('model/Y.txt.npy')

X = X.astype('float32')
X = X/255
    
test = X[3]
cv2.imshow("aa",test)
cv2.waitKey(0)
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]
Y = to_categorical(Y)

print(Y)
if os.path.exists('model/model.json'):
    with open('model/model.json', "r") as json_file:
        loaded_model_json = json_file.read()
        classifier = model_from_json(loaded_model_json)
    json_file.close()    
    classifier.load_weights("model/model_weights.h5")
    classifier._make_predict_function()       
else:
    inception = InceptionV3(input_shape = (75, 75, 3), include_top = False, weights = 'imagenet')
    inception.trainable = False
    classifier = Sequential()
    classifier.add(inception)
    classifier.add(Convolution2D(32, 3, 3, input_shape = (32, 32, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Flatten())
    classifier.add(Dense(output_dim = 256, activation = 'relu'))
    classifier.add(Dense(output_dim = Y.shape[1], activation = 'softmax'))
    print(classifier.summary())
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    hist = classifier.fit(X, Y, batch_size=16, epochs=15, shuffle=True, verbose=2)
    classifier.save_weights('model/model_weights.h5')            
    model_json = classifier.to_json()
    with open("model/model.json", "w") as json_file:
        json_file.write(model_json)
    f = open('model/history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()
    f = open('model/history.pckl', 'rb')
    data = pickle.load(f)
    f.close()
    acc = data['accuracy']
    accuracy = acc[9] * 100
    print("Training Model Accuracy = "+str(accuracy))
