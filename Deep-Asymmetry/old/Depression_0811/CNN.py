#%%
from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

#%%
train_data_dir = './fig/PCA_alpha/train'
test_data_dir = './fig/PCA_alpha/test'

#%%
'''normalization model'''
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape = (288, 288, 3)))                     
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Flatten())

model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

print(model.summary())

#%%
#binary_crossentropy ���� mse �ս� �Լ��� ������ loss function�� �� ����
model.compile(loss='binary_crossentropy',
              optimizer='Adam',
              metrics=['acc'])

########### split ���� �� �ٲٱ� ###############
#####Basic CNN �� ��� 0.3
#####Modified �� ��� 0.25
###########################################
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(288, 288),
    batch_size=5,
    class_mode='binary',
    color_mode="rgb",
    subset='training',
    seed=3
)

validation_generator = test_datagen.flow_from_directory(
    test_data_dir, # same directory as training data
    target_size=(288, 288),
    batch_size=5,
    class_mode='binary',
    color_mode="rgb",
    subset='validation',
    seed=3
)

#%%
history = model.fit_generator(
    train_generator,
    steps_per_epoch= 10,
    validation_data=validation_generator,
    validation_steps = validation_generator.samples,
    epochs=30
)

#%%
# 5. �� ���ϱ�
print("-- Evaluate --")
scores = model.evaluate_generator(validation_generator, steps=3)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))
print("%s: %.2f%%" %(model.metrics_names[0], scores[0]))
print(scores)

# 6. �� ����ϱ�f
# print("-- Predict --")
# output = model.predict_generator(validation_generator, steps=5)
# np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
# print(validation_generator.class_indices)
# print(output)

#Visualization

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc)+1)

plt.plot(epochs, acc, 'bo', label='Train Acc')
plt.plot(epochs, val_acc, 'b', label='Test Acc')
plt.title('Train and Test Accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label='Train loss')
plt.plot(epochs, val_loss, 'b', label='Test loss')
plt.title('Train and Test Loss')
plt.legend()
plt.show()

# %%
model.save('0729.h5')
# %%


# %%
