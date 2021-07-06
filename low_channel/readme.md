# 제목 : 

2020.06.08 수정

코드 확인 : 

## 1. 데이터셋 설명 

Hospital Universiti Sains Malaysia 에서 인간 윤리 위원회가 승인한 실험 설계에 따라 34명의 MㄴDD, 30명의 정상 환자 수집 MDD 환자는 DSM-IV에 따라 우울증 진단 기준 충족된 환자, 약물 효과를 피하기 위해 2주간의 약물 세척 시간을 거침

## 2. 기존 연구 리뷰



### 2.1. 논문에 대한 방법론 정리

| 논문명                                                       | 전처리 방법                                                  | 모델                                                         | 성능                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [Electroencephalogram (EEG)-based computer-aided technique to diagnose major depressive disorder (MDD)](https://www.sciencedirect.com/science/article/pii/S1746809416300866#bib0220) | 1. 노이즈 제거 (MSEC)     <br />2. 특징 추출      <br />1) EC, EO 데이터에서 2분의 epoch 추출        2) 알파 반구간 비대칭      <br />3) 뇌파 스펙트럼 파워     <br />3. 특징 세트 선택      <br />각각의 Feature들마다 AUC 값에 기반한 순위를 매겨 5,  10, 15, 19개의 feature 집단을 생성 | 1. LR     2. SVM     3. NB                                   | (1) 로지스틱 회귀      - (10개의 Feature) 반구형 알파 비대칭 방법이  97.6% 정확도, 96.66% 민감도, 특이도 =  98.5           <br />(2) NB 분류모델     - (5개의 Feature) 반구형 알파 비대칭 방법이 96.8           <br />(3) SVM     - (19개의 Feature) 반구형 알파 비대칭 방법이 98.4 |
| [A deep learning framework for automatic diagnosis of unipolar depression](https://www.sciencedirect.com/science/article/pii/S1386505619307154#!) | 1. 뇌파 노이즈 제거 (MSEC)         <br /> 2. 전처리    <br /> 1초의 window length (256 sample) 로 분할     <br />- 19개의 채널을 가지며, 256sample     (256 , 19) | 1. 1DCNN     2. 1DCNN + LSTM          Adam     Binary cross entropy 손실함수     sigmoid | 1DCNN      EO 데이터 - 19개 채널을 사용하였을 때      acc 98.32      precision = 99.78     recall = 98.34      f-score = 97.65          <br />1DCNN + LSTM     19개 채널 사용 시,     정확도 = 95.97 %,      정밀도 = 99.23 %,      재현율 = 93.67 %      f-score = 95.14 |
| [Detection of major depressive disorder using linear and non-linear features from EEG signals](https://link.springer.com/article/10.1007/s00542-018-4075-z) | 1. 뇌파 노이즈 제거      <br />0.5hz -32hz 신호만 남기고 cut     <br />ICA를 사용하여 눈 깜빡임 제거          <br />2. feature extraction      <br />1) Band Power      델타 (0.5–4Hz), 세타 (4–8Hz), 알파 (8-13Hz) 및 베타 (13-30Hz)     <br /> 2) 반구형 비대칭     <br /> 3) 웨이블릿 변환         <br /> 3. 차원 축소     비선형 분석(웨이블릿)에 있어서 PCA로 차원 축소 | (1) MLPNN      (2) RFBN      (3) LDA      (4) QDA            | 1) 밴드 파워     알파 전력이 MLPNN 에서 acc 91.67을 달성          <br />2) 반구형 비대칭     알파 비대칭에 대한 QDA에서 73.33의 분류 정확도          <br />3) 비선형 Feature     RWE : RBFN , WE : LDA에서 각각 90%의 분류 정확도          <br />4) 조합     RWE와 알파 파워의 조합은 정확도: 93.33  민감도 : 94.44 특이도 87.78을 보여주었다 |
| [Classification of Depression Patients and Normal Subjects Based on Electroencephalogram (EEG) Signal Using Alpha Power and Theta Asymmetry](https://link.springer.com/article/10.1007/s10916-019-1486-z#Bib1) | 1. 뇌파 노이즈 제거      <br />0.5hz~32hz 신호만 남기고 cut    <br /> ICA를 사용하여 눈 깜빡임 제거     <br />     2. feature extraction      <br />1) Band Power      <br />델타 (0.5–4Hz), 세타 (4–8Hz), 베타  (13–30Hz)     알파1 (8~10.5), 알파2(10.5~13)     <br /> 2) 반구간 세타 비대칭     - 8개의 feature          <br />3. 데이터 메트릭스 구성    <br /> 1) 각 밴드파워에 대한 세트     (60, 19) * 5band         <br /> 2) 알파+세타 비대칭 조합을 위한 세트 (60*27)         <br /> 3. Feature Selection     MCFS 를 사용해 피쳐 선택 | (1) 로지스틱 회귀     (2) SVM      (3) NB     (4) DT         | 2) 대역 전력 기반     알파 전력 기반 SVM에서 84.50의  ACC     (          <br />3) 조합     알파2 파워와 세타 비대칭의 조합 - SVM에서 88.33의 분류 정확도     ( 특이도 89.41 %      민감도 90.81 %) |
| Imagery Signal-based Deep  Learning Method for Prescreening Major Depressive Disorder | 1. Feature Selection     저채널 장비에서 사용되는 Fp1, Fp2 채널만 사용          <br />2. STFT     <br />3. 스펙트로그램 변환<br />     4. 2채널 스펙트로그램 병합     <br />5. 차원 축소 - 기존 이미지 1/10 | CNN 모델                                                     | val_acc = 0.75      <br />val_loss = 0.3                     |
| Automated Depression Detection Using Deep Representation and Sequence Learning with EEG Signals | 특별한 전처리 없이  raw  신호를 딥러닝 모델에 바로 넣음      | CNN  모델 + LSTM                                             | 98.98                                                        |
| Detection of Depression and Scaling of Severity Using Six Channel EEG Data | \1. 노이즈 제거<br/> \2. 6개의 채널 (FT7, FT8, T7, T8, TP7, TP8) 사용<br/> \3. 2초의 epoch 데이터로 분할하고 2초 데이터의 피크 진폭을 측정하고 표준편차의 3배보다 높은 피크 진폭의 에포크는 삭제<br/> \4. 특징 추출 : 대역 전력 <br/> \5. 반구간 비대칭 계산<br/> \6. 비선형 특징 <br/> \- 샘플 엔트로피<br/> \- DFA | SVM                                                          | 90.26                                                        |

## 4. 본 논문에서 사용할 방법론

![](https://github.com/ark1st/MDD/blob/master/diagram/mdd-diagram.jpg?raw=true)



#### 4.1. Noise Reduction

전력선 노이즈, 근육 움직임 등의 인공물들은 50Hz 이상의 주파수를 가짐. Low-Pass Filter와 High pass filter를 사용하여 0.5Hz~32Hz의 주파수만 남기고 cut함.

#### 4.2. Feature extraction

Band pass filter를 사용하여 원하는 주파수 대역대만 추출. 1개의 채널(공간)당 4개의 밴드대역대가 존재하게 됨.

![](https://github.com/ark1st/MDD/blob/master/diagram/wave.png?raw=true)

#### 4.3. Data Normalization

데이터 세트의 스케일을 맞추기 위해서 data normalization 작업을 시행함. train set 에서 fit 된 scaler로  train set과 test set을 정규화함.



#### 4.4. 딥러닝 모델

2D Convolution layer를 사용한 딥 러닝 모델을 사용하였다. 

2층의 Conv2D 레이어 및 ReLU Activation Function를 사용하였다. 

모델 학습에는 Adam Optimizer를 사용하였다.

![](https://github.com/ark1st/MDD/blob/master/diagram/model.PNG?raw=true)

#### 4.5. 성능평가

confusion matrix에 기반하여 도출된 Accuracy, Sensitivity, specificity을 사용하여 성능평가.

![Understanding Confusion Matrix - Towards Data Science](https://miro.medium.com/max/712/1*Z54JgbS4DUwWSknhDCvNTQ.png)



#### 5. 결과

train-test accuracy loss graph

fig에서 Epoch가 증가함에 따라 Accuracy가 증가하고 loss가 감소하는 것으로 나타났다. 고안된 모델에서 94.4의 Test accuracy를 나타났다.

![](https://github.com/ark1st/MDD/blob/master/diagram/train-test.PNG?raw=true)

10*10-fold cross validation 결과

```
Acc : 96.80 (+/- 0.07%)
Sen : 90.38 (+/- 0.22%)
Spec : 99.00 (+/- 0.10%)
```



#### 6. 토론

기존 연구와 비교

| 논문명                                                       | Dataset                               | Feature extraction method             | Channel  selection             | 분류기        | 성능(Accuracy) |
| ------------------------------------------------------------ | ------------------------------------- | ------------------------------------- | ------------------------------ | ------------- | -------------- |
| [Electroencephalogram (EEG)-based computer-aided technique to diagnose major depressive disorder (MDD)](https://www.sciencedirect.com/science/article/pii/S1746809416300866#bib0220) | Wajid Mumtaz                          | + EEG alpha asymmetry                 | 19                             | SVM           | 98.4           |
| [A deep learning framework for automatic diagnosis of unipolar depression](https://www.sciencedirect.com/science/article/pii/S1386505619307154#!) | Wajid Mumtaz                          | 1s (256 samples) segmented  EEG Data  | 19                             | CNN (1D)      | 98.32          |
| [Detection of major depressive disorder using linear and non-linear features from EEG signals](https://link.springer.com/article/10.1007/s00542-018-4075-z) | Wajid Mumtaz                          | Linear Features + Non-linear Features | 19                             | MLPNN, RBFN   | 93.33          |
| [Classification of Depression Patients and Normal Subjects Based on Electroencephalogram (EEG) Signal Using Alpha Power and Theta Asymmetry](https://link.springer.com/article/10.1007/s10916-019-1486-z#Bib1) | Wajid Mumtaz                          | Alpha 2 power + theta symmetry        | 19                             | SVM           | 88.33          |
| Deep Learning based Pre-screening method for Depression with Imagery Frontal EEG Channels | Wajid Mumtaz                          | Imagery Frontal EEG Channels (STFT)   | 2 (Fp1, Fp2)                   | VGG16         | 87.5           |
| Automated Depression Detection Using Deep Representation and Sequence Learning with EEG Signals | Acharya et al.                        | Raw EEG Signals                       | 4 (Fp1, Fp2, T3, T4)           | CNN + LSTM    | 99.12, 97.66   |
| Detection of Depression and Scaling of Severity Using Six Channel EEG Data | Central Institute of Psychiatry (CIP) | Linear Features + Non-linear Features | 6 (FT7, FT8, T7, T8, TP7, TP8) | SVM (ReliefF) | 96.02          |
| Present Study                                                | Wajid Mumtaz                          | Band power                            | 4 (Fp1, Fp2, F7, F8)           | CNN (Conv2D)  | 96.80          |

## 



딥러닝 학습과 검증에 있어서 데이터의 절대적 양은 아주 중요한 요소. but 현재 데이터의 양으로는 모델의 학습과 평가에 있어서 한계가 있다고 판단됨. 

따라서, 더 많은 양의 데이터를 확보하거나, 데이터의 양적 한계를 극복하는 방법론을 고안하여, 더 좋은 성과를 보여줄 수 있도록.

#### 7. 결론

뇌파 데이터에서 4개의 전두엽 채널을 선택하고, 이를 밴드 대역 주파수별로 특징 추출한 데이터를 2차원의 Convolution laver 기반의 딥러닝 모델에 학습 시켜서 96.8의 결과를 얻음. 저채널 기기를 사용한 뇌파 사전 진단에 있어서 밴드 대역 주파수를 사용한 특징 추출 방법과 2차원의 컨볼루션 레이어를 사용한 딥러닝 모델은 효과적이다.