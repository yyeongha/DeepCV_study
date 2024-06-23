# A ConvNet for the 2020s

내 블로그에 썼었던 리뷰를 그대로 가져옴.

내 블로그 : [블로그리뷰](https://yyeongha.github.io/posts/ConvNet/)

논문 출처 : [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545v2)

2020년도에 ViT(Vision Transofrmer)가 발표된 이후 Vision task에서 Transformer에 연구가 집중되고 있지만 CNN에 Transformer 구조 및 최신 기법들을 적용한 ConvNeXt라는 모델을 제안하고 있으며 높은 성능을 통해 CNN이 여전히 강하다는 것을 주장하는 논문이다.


# 1. Introduction
2010년대 컴퓨터 비전 분야는 이미지 인식에 탁월한 ConvNets의 발전으로 급성장했다. 

자연어 처리 분야의 ViT(Vision Transformer)가 컴퓨터 비전에 도입되면서 새로운 변화를 맞이했다. ViT는 이미지 분류에는 뛰어났지만 객체 감지나 이미지 분할과 같은 작업에는 한계를 보였다. 

이후 Swin Transformer와 같은 계층적 트랜스포머 모델이 등장하여 ConvNets의 장점을 일부 차용하며 다양한 컴퓨터 비전 작업에서 좋은 성능을 보였다. 

본 논문에서는 ConvNets를 개선하여 트랜스포머 모델만큼의 성능을 내는 ConvNeXt라는 새로운 모델을 제안한다. ConvNeXt는 Swin Transformer와 비슷하거나 더 나은 성능을 보였고, 기존 ConvNets의 단순함과 효율성도 유지했다. 본 연구는 ConvNets가 여전히 컴퓨터 비전 분야에서 중요한 역할을 할 수 있음을 보여주며, 컴퓨터 비전 모델 설계에 새로운 시각을 제시한다. 

![figure1](https://github.com/yyeongha/yyeongha.github.io/blob/main/assets/img/favicons/2024-05-15-convnet-img/figure1.png?raw=true) 

중간 선을 기준으로 ImageNet-1K, ImageNet-22K 데이터를 통해서 학습한 모델의 top1 accuracy이다.

각 버블의 면적은 모델 제품군의 변형 FLOP(연산량)에 비례한다. 


# 2. Modernizing a ConvNet: a Roadmap
본 논문에서는 ResNet-50/200 모델을 시작으로 단계별로 모델 구조를 변경하며 성능을 개선하는 과정을 거쳐, 최종적으로 ConvNeXt라는 새로운 ConvNet 모델을 제안한다. 각 단계는 다음과 같다.

![figure2](https://github.com/yyeongha/yyeongha.github.io/blob/main/assets/img/favicons/2024-05-15-convnet-img/figure2.png?raw=true)

## 1) ResNet-50/200 
resnet 50/200을 베이스로 잡아 transformer의 학습기법을 도입
* epoch을 90 -> 300으로 증가
* AdamW optimizer 사용
* data augmentation(Mixup, Cutmix, RandAugment, Random Erasing) 추가
* Regularization(Stochastic Depth, Label Smoothing.)
-> accuracy : 76.1% -> 78.8% 

이를 통해 전통적인 ConvNet과 ViT의 성능 차이의 많은 부분이 training technques에서 기인하였음을 알 수 있다. 


## 2) macro design
* Changing stage compute ratio

ResNet-50 모델은 각 스테이지마다 블록이 3, 4, 6, 3개씩 구성되어 있는데, 이를 Swin Transformer의 스테이지별 블록 비율인 1:1:3:1에 맞춰 3, 3, 9, 3개로 조정했다.

![macro](https://github.com/yyeongha/yyeongha.github.io/blob/main/assets/img/favicons/2024-05-15-convnet-img/macro.png?raw=true)

-> accuracy : 78.8% -> 79.4%

* Changing stem to “Patchify”

기존 모델들의 첫 Conv 레이어(Stem)부분을 보면
  * ResNet : stride2, 7x7 conv layer,max pool을 통해 이미지를 4x4로 downsampling
  * ViT : “Patchify”라는 전략을 이용한다. 이미지를 작은 패치로 나누어 각각을 개별 입력 토큰으로 취급하는데, 이 과정에서는 보통 14x14 또는 16x16 크기의 큰 커널과 비중첩(non-overlapping) 컨볼루션을 사용하여 패치를 생성한다.
    * Swin Transformer : 더 작은 4x4형태의 non-overlapping convolution을 사용하여 패치 생성

본 논문에서는 ResNet 구조를 기반으로 Swin Transformer의 방식을 적용하여, 기존의 stem cell 부분을 4x4 커널 크기의 non-overlapping convolution 레이어로 대체하는 방법을 사용했다.

![patchify](https://github.com/yyeongha/yyeongha.github.io/blob/main/assets/img/favicons/2024-05-15-convnet-img/patchify.png?raw=true)

-> accuracy : 79.4% -> 79.5%

## 3) ResNeXt
* group convolution 도입
  * 기본 ResNet 구조에서 3x3 convolution layer를 group convolution으로 대체한다. 이는 각 컨볼루션 필터가 입력 채널을 여러 그룹으로 나누어 병렬로 적용된다.
  * group convolution을 사용함으로써, FLOPs를 크게 줄이면서 Accuracy는 최대한 유지할 수 있다.

  ![resnext](https://github.com/yyeongha/yyeongha.github.io/blob/main/assets/img/favicons/2024-05-15-convnet-img/resnext.png?raw=true)

* depthwise convolution 사용
  *  group 개수를 channel수만큼 만들어 MobileNet에서 사용했던 Depthwise convolution을 적용하여 FLOPs를 대폭 줄이게 된다. 
  ![depth](https://github.com/yyeongha/yyeongha.github.io/blob/main/assets/img/favicons/2024-05-15-convnet-img/depth.png?raw=true)

* 네트워크 width 증가
  * depthwise convolution으로 인해 FLOPs이 감소하기 때문에, 이를 보완하기 위해 네트워크의 폭(width: 64->96)을 증가시킨다. 이는 모델의 용량을 증가시켜 성능을 향상시키는 역할을 한다.

-> accuracy : 79.5% -> 80.5%

## 4) Inverted Bottleneck
* Transformer의 MLP 블록이 중간에서 채널을 4배로 늘렸다 줄이는 Inverted Bottleneck 구조를 ConvNeXt에 적용한다.
* inverted bottleneck을 적용하면 downsampling residual block의 1x1 convolution layer에서 채널 수가 감소하기 때문에 FLOPs을 크게 줄일 수 있다.

![figure3](https://github.com/yyeongha/yyeongha.github.io/blob/main/assets/img/favicons/2024-05-15-convnet-img/figure3.png?raw=true)
(a) ResNeXt (Bottleneck 구조) \
(b) Inverted Bottleneck \
(c) Spatial Depthwise Conv layer 위치 이동

![figure3_2](https://github.com/yyeongha/yyeongha.github.io/blob/main/assets/img/favicons/2024-05-15-convnet-img/figure3_2.png?raw=true)

-> accuracy : 80.5% -> 80.6%


## 5) 큰 커널 크기
전통적인 ConvNet에서는 작은커널(예:3x3)을 여러 층에 걸쳐 쌓는 방식이 일반적이었는데, 이는 하드웨어 구현이 효율적이고 계산비용이 낮기 때문이다.

ViT에서는 각 층에서 global receptive field를 갖는 non local self attention를 사용한다. 이로인해 각 층에서 더 넓은 영역의 정보를 처리할 수 있게 된다.

Swin Transformer는 self-attention block을 도입하여 최소 7x7 크기의 윈도우를 사용한다. 이를 통해 각층이 더 큰 receptive field를 갖게 된다.

* depthwise convolution layer 위치 이동
  * 큰 커널을 사용하기 위한 전제조건으로, depthwise convolution layer의 위치를 상위 레이어로 이동시켰다. 이는 Transformer 설계에서 영감을 받은 것으로 복잡하거나 비효울적인 모듈(MSA, large-kernel conv)은 채널수가 적은 상위레이어에서 처리되도록 한다.

![위치이동](https://github.com/yyeongha/yyeongha.github.io/blob/main/assets/img/favicons/2024-05-15-convnet-img/%EC%9C%84%EC%B9%98%EC%9D%B4%EB%8F%99.png?raw=true)

-> accuracy : 80.6% -> 79.9%

* 큰 커널 크기 적용 실험 (3x3 -> 7x7)
  * 논문에서는 다양한 커널크기(3, 5, 7, 9, 11)를 실험하여 ConvNet의 성능에 미치는 영향을 평가했다.
  * 3x3 커널에서는 79.9%의 정확도를 기록한 반면, 7x7 커널에서는 80.6%로 성능이 향상되었다. 이는 네트워크의 FLOPs가 거의 동일하게 유지되었음을 의미한다.

-> accuracy : 79.9% -> 80.6%

## 6) micro design
* BERT와 GPT-2 이후 Transformer 모델들에서 ReLU 대신 GELU 활성화 함수를 주로 채택하는 추세에 맞춰 convnext 또한 GELU를 사용하도록 변경하였다.

![relugelu](https://github.com/yyeongha/yyeongha.github.io/blob/main/assets/img/favicons/2024-05-15-convnet-img/relugelu.png?raw=true) 

* activation functions(활성화 함수) 개수 줄임
  * Transformer 블록은 각 층에서 활성화 함수를 최소한으로 사용하는 전략을 따르는데, ConvNeXt도 이러한 접근 방식을 차용하여 두 개의 1x1 컨볼루션 레이어 사이에 GELU 활성화 함수를 하나만 남겼다.

* normalization layers(정규화 레이어)의 개수 줄임
  * Transformer 블록은 정규화 레이어도 최소한으로 사용하는 경향을 보인다. 이러한 특징을 참고하여 ResNet 블록에서 두 개의 BatchNorm(BN) 레이어를 제거하고, 1x1 컨볼루션 레이어 앞에 하나의 BN 레이어만 남겼다.

* Batch Normalization(BN)을 Layer Normalization(LN)으로 변경
  * Batch Normalization(BN)은 ConvNet에서 일반적으로 사용되는 정규화방법이지만, 여러가지 복잡성을 가지고 있어 성능에 부정적인 영향을 미칠 수 있다.
  * Transformer에서는 Layer Normalization(LN)을 사용하며, 이를 ConvNeXt에 적용하였다.

![bntoln](https://github.com/yyeongha/yyeongha.github.io/blob/main/assets/img/favicons/2024-05-15-convnet-img/bntoln.png?raw=true)

* 별도의 다운샘플링 레이어
  * ResNet에서는 각 스테이지의 시작부분에서 3x3 convolution과 stride2를 사용하여 공간적 다운 샘플링을 수행한다.
  * 하지만 Swin Transformer에서는 2x2 convolution 레이어를 사용하여 공간적 다운 샘플링 레이어를 추가한다.
  * 이를 ConvNeXt에도 적용하여, 2x2 convolution 레이어를 사용하여 공간적 다운샘플링을 수행하였다. 이 과정에서 normalization layer를 추가하여 훈련을 안정화시켰다.

이러한 단계를 거쳐 개발된 ConvNeXt 모델은 Swin Transformer와 유사한 FLOPs, 파라미터 수, 처리량, 메모리 사용량을 가지면서도, Swin Transformer에 필요한 특별한 모듈 없이도 뛰어난 성능을 보인다.

![convnext](https://github.com/yyeongha/yyeongha.github.io/blob/main/assets/img/favicons/2024-05-15-convnet-img/convnext.png?raw=true)


# 3. Experiments
![experiment](https://github.com/yyeongha/yyeongha.github.io/blob/main/assets/img/favicons/2024-05-15-convnet-img/experiment.png?raw=true)
* ViT, Swin과 같이 다양한 사이즈의 모델들을 구성했다.
* 가운데는 output의 채널 수, 마지막은 block의 수
블록의 수는 1:1:3:1의 비율을 가지고 있다.

## 1) classification
![classification](https://github.com/yyeongha/yyeongha.github.io/blob/main/assets/img/favicons/2024-05-15-convnet-img/classification.png?raw=true)

## 2) object detection, segmentation
![ods](https://github.com/yyeongha/yyeongha.github.io/blob/main/assets/img/favicons/2024-05-15-convnet-img/ods.png?raw=true)


# 4. Conclusions
* 순수 ConvNet: ConvNeXt는 순수한 ConvNet 구조를 기반으로 하여, Transformer 기반의 특화된 모듈 없이도 높은 성능을 발휘한다.
* 효율성: ConvNeXt는 Swin Transformer와 비교하여 더 낮은 FLOPs를 유지하면서도 경쟁력 있는 성능을 보인다.
* 실용성: 이러한 특성 덕분에 ConvNeXt는 다양한 환경에서 더 효율적이고 실용적으로 사용할 수 있다.

---