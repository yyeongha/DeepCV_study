# SPP-Net: Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition

논문 출처 : [SPP-Net]()

## 1. 서론
기존의 딥 컨볼루션 신경망(CNN)은 고정된 입력 이미지 크기(예: 224x224)만을 입력으로 받아들였다. 밑의 figure 1을 보면 알 수 있듯이, 완전 연결 계층(fc layer)이 고정된 크기의 벡터만 입력으로 받기 때문에 CNN은 입력 이미지를 고정된 크기에 맞춰주기 위해 이미지를 잘라내는 크롭(crop)이나 왜곡하는 워프(warp)와 같은 방법을 사용해야 했다. \
하지만 크롭을 사용하면 잘려나간 부분의 이미지 정보가 손실되어 전체 이미지를 온전히 반영하지 못했고, 워프를 사용하는 경우에는 이미지가 왜곡되어 변형이 발생했다. 자동차 이미지에 크롭을 적용하면 자동차의 일부분만 CNN을 통과하게 되고, 등대 이미지에 워프를 적용하면 등대가 좌우로 퍼진 형태로 CNN을 통과하게 된다.

이러한 요구사항은 임의의 크기/척도를 가진 이미지나 하위 이미지에 대해 인식 정확도를 떨어뜨릴 수 있다는 문제점이 있다. 본 논문에서는 이러한 문제를 해결하기 위해 "공간 피라미드 풀링(spatial pyramid pooling, SPP)"이라는 새로운 풀링 전략을 CNN에 도입하는 SPP-Net을 제안한다.

![figure1]()


## 2. SPP-Net의 핵심 아이디어: 공간 피라미드 풀링(SPP)

SPP는 기존 CNN 모델에서 사용되던 고정된 크기의 풀링 대신, 다양한 크기의 풀링 영역을 사용하여 특징 맵을 추출하는 방법이다. 이를 통해 입력 이미지의 크기가 달라져도 고정된 크기의 특징 벡터를 얻을 수 있게 된다. 

SPP는 다음과 같은 단계로 이루어진다.

1.  **이미지 분할:** 입력 이미지를 여러 개의 겹치지 않는 영역으로 분할한다. 이때 분할되는 영역의 크기는 피라미드 형태로 구성된다. 즉, 작은 영역부터 큰 영역까지 다양한 크기의 영역으로 이미지를 분할한다.
   
2.  **각 영역별 풀링:** 각 영역에서 풀링(pooling) 연산을 수행하여 해당 영역의 특징을 추출한다. 일반적으로 최대 풀링(max pooling)을 사용하여 각 영역에서 가장 큰 값을 선택한다.
   
3.  **특징 벡터 생성:** 각 영역에서 추출된 특징들을 연결하여 고정된 크기의 특징 벡터를 생성한다. 이렇게 생성된 특징 벡터는 입력 이미지의 크기와 상관없이 항상 동일한 크기를 가지므로, 이후 완전 연결 계층에 입력으로 사용될 수 있다.


아래 그림은 SPP의 과정을 시각적으로 보여준다.
![figure3]()

## 3. SPP-Net의 장점

SPP-Net은 기존 CNN 모델에 비해 다음과 같은 장점을 가진다.

1.  **다양한 크기의 이미지 처리 가능:** SPP를 통해 입력 이미지의 크기나 비율에 제약을 받지 않고 다양한 크기의 이미지를 처리할 수 있다. 기존 CNN 모델은 고정된 입력 크기만을 허용하여 입력 이미지를 잘라내거나 변형해야 했지만, SPP-Net은 이러한 제약 없이 다양한 크기의 이미지를 입력으로 받아 효율적으로 처리할 수 있다.
   
2.  **객체 변형에 대한 강건성:** 다양한 크기의 풀링 영역을 사용하여 특징을 추출하므로, 객체의 크기나 위치 변화에 강건한 특징 표현을 얻을 수 있다. 기존 CNN 모델은 객체의 크기나 위치가 변하면 인식 성능이 저하되는 문제가 있었지만, SPP-Net은 다양한 크기의 풀링 영역을 사용하여 이러한 문제를 완화했다.
   
3.  **높은 인식 정확도:** SPP-Net은 이미지 분류 및 객체 감지 작업에서 기존 CNN 모델보다 높은 인식 정확도를 보인다. SPP를 통해 추출된 특징 벡터는 객체의 다양한 크기와 위치 정보를 반영하여 더욱 풍부한 정보를 제공하며, 이는 인식 정확도 향상으로 이어진다.
   
4.  **객체 감지 속도 향상:** SPP-Net은 객체 감지 과정에서 불필요한 연산을 줄여 기존 방법보다 훨씬 빠른 속도로 객체를 감지할 수 있다. 기존의 R-CNN과 같은 방법은 각 후보 영역마다 CNN을 반복적으로 적용해야 했지만, SPP-Net은 전체 이미지에 대해 한 번만 CNN을 적용하고 SPP를 사용하여 각 후보 영역의 특징을 추출하므로 연산량을 크게 줄일 수 있다.
   

## 4. 실험 결과

SPP-Net의 성능을 검증하기 위해 ImageNet 2012, Pascal VOC 2007, Caltech101 데이터셋을 사용하여 이미지 분류 및 객체 감지 실험을 수행했다.

*   **ImageNet 2012:** 다양한 CNN 아키텍처에 SPP를 적용하여 기존 모델보다 높은 성능을 달성했다. 특히, 단일 전체 이미지 표현만으로도 최첨단 분류 결과를 얻었다.
   
*   **Pascal VOC 2007 및 Caltech101:** 미세 조정 없이 단일 전체 이미지 표현을 사용하여 최첨단 분류 결과를 달성했다.
   
*   **객체 감지:** SPP-Net을 객체 감지에 적용하여 R-CNN보다 24-102배 빠른 속도로 객체를 감지하면서도 더 우수하거나 비슷한 정확도를 달성했다.
   

## 5. SPP-net의 한계점 및 후속 연구

SPP-net은 객체 탐지 분야에 큰 영향을 미쳤지만, 다음과 같은 한계점을 가지고 있다.

*   **End-to-end 학습 불가:** SPP layer 이전의 CNN Feature Extractor와 이후의 SVM 또는 FC layer가 각각 독립적으로 학습되므로, end-to-end 학습이 불가능하다.
*   **Region proposal 생성 방식:** Selective Search와 같은 전통적인 region proposal 생성 방식을 사용하므로, region proposal 생성 과정이 객체 탐지 속도를 저하시킨다.
이러한 한계점을 극복하기 위해 Fast R-CNN, Faster R-CNN 등 후속 연구들이 등장했다. 특히 Faster R-CNN은 Region Proposal Network (RPN)를 도입하여 region proposal 생성 과정을 CNN 모델 내부로 통합하고, end-to-end 학습을 가능하게 함으로써 객체 탐지 성능과 속도를 크게 향상시켰다.

## 5. 결론

SPP-Net은 기존 CNN 모델의 한계를 극복하고 다양한 크기의 이미지를 효과적으로 처리할 수 있는 새로운 가능성을 제시했다. SPP를 통해 객체 변형에 강건하고 높은 인식 정확도를 달성할 수 있으며, 특히 객체 감지 분야에서 획기적인 속도 향상을 이끌어냈다. SPP-Net은 딥러닝 기반 컴퓨터 비전 시스템의 발전에 크게 기여했으며, 다양한 실제 응용 분야에서 딥러닝 모델의 활용성을 높이는 데 기여할 것으로 기대된다.