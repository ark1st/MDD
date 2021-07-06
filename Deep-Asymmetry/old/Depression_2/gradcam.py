#%%

import keras
import os, shutil
import numpy as pd
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
import cv2

# %%
from keras.models import load_model
model = load_model('0728.h5')

model.summary()
# %%
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
folder_path = '../test_set'
# 이미지 경로
img_path = folder_path + '/H/H (1).png'

img = image.load_img(img_path, target_size=(288, 288))

# (224, 224, 3) 크기의 넘파이 float32 배열
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

# %%
from keras.preprocessing.image import ImageDataGenerator

# 모든 이미지를 1/255로 스케일을 조정합니다
test_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = test_datagen.flow_from_directory(
        folder_path,
        target_size=(288, 288),
        batch_size=20,
        class_mode='binary')
validation_generator

# %%
img, label = validation_generator.next()
X=[]
y=[]
for i in range(10):
    #print(i, ":", label[i])
    #plt.imshow(img[i])
    #plt.show()
    X.append(img[i].reshape((1,) + img[i].shape))
    y.append(label[i])

# %%
def create_heatmap(input_image, pred,k):
    max_arg= np.argmax(pred[0])
    output = model.output[:, max_arg]
    last_conv_layer= model.get_layer('conv2d_30')
    grads = K.gradients(output, last_conv_layer.output)[0]


    pooled_grads = K.mean(grads, axis=(0, 1, 2))

    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])


    pooled_grads_value, conv_layer_output_value = iterate([input_image])

    for i in range(128):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    heatmap = np.mean(conv_layer_output_value, axis=-1)
    
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    # cv2 모듈을 사용해 원본 이미지를 로드합니다
    img = input_image.reshape(288,288,3)
    img = img*255
    #img = cv2.imread('10_test/cats/6.jpg')
    # heatmap을 원본 이미지 크기에 맞게 변경합니다
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # heatmap을 RGB 포맷으로 변환합니다
    heatmap = np.uint8(255 * heatmap)

    # 히트맵으로 변환합니다
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    cv2.imwrite('heatmap/heatmap_'+str(k)+'.jpg', heatmap) #히트맵 저장
    
    #image Grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # 0.4는 히트맵의 강도입니다
    superimposed_img = heatmap * 0.4 + img

    # 디스크에 이미지를 저장합니다
    cv2.imwrite('heatmap/heatmap_img_'+str(k)+'.jpg', superimposed_img)
    plt.imshow(superimposed_img/255)
    plt.show()

# %%

o=0
x=0
for i in range(10):
    print("{} 번째 예측".format(i))
    plt.imshow(X[i].reshape(288,288,3))
    plt.show()
    pred = model.predict(X[i])
    print("이미지 Label : {}".format(y[i]))
    print("예측한 Label : {}, ({})".format(np.round(pred), pred))
    if np.round(pred) == y[i]: 
        print("예측이 맞았습니다.") 
        o+=1
    else: 
        print("예측이 틀렸습니다")
        x+=1
    create_heatmap(X[i], pred, i)
    print("======================")
print("맞은 개수 : {}, 틀린 개수 : {}".format(o,x))

# %%

# %%
