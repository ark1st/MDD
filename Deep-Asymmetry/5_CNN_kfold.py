#%%
'''
data_dir = [
    'D:\Depression\MDD_dataset\H\plot']#, # 0 : delta
#    'D:/Depression/plot/Nor_ICA_name/'+'theta', # 1 : theta
#    'D:/Depression/plot/Nor_ICA_name/'+'alpha', # 2 : alpha
#    'D:/Depression/plot/Nor_ICA_name/'+'beta', # 3 : beta
#]
H_filelist = [0,0,0,0]
MDD_filelist = [0,0,0,0]
H_label = [0,0,0,0]
MDD_label = [0,0,0,0]

H_filelist[0] = os.listdir(data_dir[0]+"/H")
H_label[0] = list(np.zeros(len(H_filelist[0])))

MDD_filelist[0] = os.listdir(data_dir[0]+"/MDD")
MDD_label[0] = list(np.ones(len(MDD_filelist[0])))

i = 0
id_Series=pd.Series(H_filelist[i]+MDD_filelist[i])
label_Series= pd.Series(H_label[i]+MDD_label[i])
label_Series = label_Series.astype(str)
df = pd.DataFrame({'id' : id_Series,'label' :label_Series})
df.to_csv("id-label.csv")
'''
#%%
from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras import backend as K
import os
import pandas as pd
import numpy as np
#%%
def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())


def true_pred(model, validation_generator):
    validation_generator.reset()
    predictions = model.predict_generator(validation_generator, steps=1)
    test_preds = np.rint(predictions)
    l=test_preds.shape[0]
    test_trues = validation_generator.classes
    return test_trues, test_preds
# %%
##############################
####### Change Path ##########
##############################
df = pd.read_csv("id-label.csv", index_col='index') #load dictionary dataframe
df['label'] = df['label'].astype(int)
df['label'] = df['label'].astype(str)
from sklearn.model_selection import train_test_split
data_index = [i for i in range(4275)]

#%%
def CNN_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape = (64, 64, 3)))                     
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

#%%
def generate_Image(train, test, file_dir):
    datagen = ImageDataGenerator(rescale=1./255)
    train_generator = datagen.flow_from_dataframe (
        dataframe=train,
        directory=file_dir,
        x_col='id',
        y_col='label',
        batch_size=75,
        seed = 0,
        class_mode='binary',
        target_size=(64,64),
        shuffle=True

    )

    validation_generator = datagen.flow_from_dataframe (
        dataframe=test,
        directory=file_dir,
        x_col='id',
        y_col='label',
        batch_size=915,
        seed = 0,
        class_mode='binary',
        target_size=(64,64),
        shuffle=False

    )
    return train_generator, validation_generator

# %%
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
# %%
scores = [0,0,0,0,0]
history = [0,0,0,0,0]
preds = [0,0,0,0,0]
trues = [0,0,0,0,0]
from keras import optimizers

##############################
####### Change Path ##########
##############################
file_dir = 'D:/Depression/MDD_dataset/H/all'
opt = optimizers.Adam(learning_rate=0.0001)

for i, (idx_train, idx_test) in enumerate(cv.split(df['id'], df['label'])):
    df_train = df.iloc[idx_train]
    df_test = df.iloc[idx_test]

    train_generator , validation_generator = generate_Image(df_train, df_test, file_dir)

    model = CNN_model()

    model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['acc', sensitivity, specificity])
              
    history[i] = model.fit_generator(
    train_generator,
    steps_per_epoch= 57,
    validation_data=validation_generator,
    validation_steps =1,
    epochs=10,
    callbacks=[],
    verbose = 0
    )

    scores[i] = model.evaluate_generator(validation_generator)

    print("%s: %.4f%%" % (model.metrics_names[1], scores[i][1]))
    print("%s: %.4f%%" % (model.metrics_names[2], scores[i][2]))
    print("%s: %.4f%%" % (model.metrics_names[3], scores[i][3]))

    trues[i], preds[i] = true_pred(model, validation_generator)

    


# %%
acc = []
sen = []
spec = []
for i in range(5):
    acc.append(scores[i][1])
    sen.append(scores[i][2])
    spec.append(scores[i][3])
# %%
print("Accucray : " + str(np.mean(acc)))
print("Sensitivity : " + str(np.mean(sen)))
print("specificity : " + str(np.mean(spec)))

# %%
from scipy import interp
from sklearn.metrics import roc_curve,auc
tprs = []
aucs = []
mean_fpr = np.linspace(0,1,100)
for i in range(5):
    fpr, tpr, t = roc_curve(trues[i], np.rint(preds[i]))
    tprs.append(interp(mean_fpr, fpr, tpr))
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.4f)' % (i, roc_auc))
plt.plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'black')
mean_tpr = np.mean(tprs, axis=0)
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='blue',
         label=r'Mean ROC (AUC = %0.4f )' % (mean_auc),lw=2, alpha=1)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")
plt.show()
# %%
len(trues[0])
len(preds[0])
# %%
# %%
