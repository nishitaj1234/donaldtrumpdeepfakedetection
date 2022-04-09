import pandas as pd
import numpy as np
import sklearn.model_selection
import keras

from keras.models import load_model

originalfile=pd.read_csv("trumporiginal.txt")
deepfakefile=pd.read_csv("deepfaketrump.txt")
deepfakefile2=pd.read_csv("deepfaketrump2.txt")
deepfakefile3=pd.read_csv("deepfaketrump3.txt")

originaldata=np.array(originalfile.drop("class",1))
originallabels=np.array(originalfile["class"])
print(len(originaldata),len(originallabels))

deepfakedata=np.array(deepfakefile.drop("class",1))
deepfakelabels=np.array(deepfakefile["class"])

deepfakedata2=np.array(deepfakefile2.drop("class",1))
deepfakelabels2=np.array(deepfakefile2["class"])

deepfakedata3=np.array(deepfakefile3.drop("class",1))
deepfakelabels3=np.array(deepfakefile3["class"])
'''print(originallabels)
print(deepfakelabels)
print(deepfakelabels2)'''

alldeepfakelabels=np.concatenate([deepfakelabels,deepfakelabels2,deepfakelabels3])
alldeepfakedata=np.concatenate([deepfakedata,deepfakedata2,deepfakedata3])

print(len(alldeepfakelabels),len(alldeepfakedata))
data=np.concatenate([alldeepfakedata,originaldata])
labels=np.concatenate([alldeepfakelabels,originallabels])

print(len(labels))

print(len(data))
print(data.shape)


#generate another deepfake based on given video. include in code, variable called all data and all labels
Xtraindata, Xtestdata, Ytraindata, Ytestdata=sklearn.model_selection.train_test_split(data,labels,test_size=0.2)


                        
                        
model=keras.Sequential([#keras.layers.Flatten(input_shape=(137,)),
                        keras.layers.Dense(128,activation='relu'),
                        keras.layers.Dense(128,activation='relu'),
                        keras.layers.Dense(128,activation='relu'),
                        keras.layers.Dense(2,activation='softmax')])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

model.fit(Xtraindata,Ytraindata,epochs=15)

model.save('deepfake.h5')




