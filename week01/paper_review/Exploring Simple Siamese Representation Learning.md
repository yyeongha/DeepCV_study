# SimSiam: Exploring Simple Siamese Representation Learning

논문 출처: [SimSiam: Exploring Simple Siamese Representation Learning](https://arxiv.org/pdf/2011.10566)

## 1. Introduction

Self-supervised learning에서는 보통 Contrastive Learning 구조를 많이 이용한다. 이러한 구조에서는 모든 output이 같은 값을 가지는 collapsing 문제를 해결하기 위해 다양한 기법들이 사용된다. 대표적으로 SimCLR와 MoCo는 negative sample을 사용하여 많은 데이터를 필요로 한다. 반면, BYOL은 negative sample이 필요 없는 방식을 제안하였다. SimSiam은 **negative pair 없이 positive pair 만을 사용**하여 학습을 진행하며, collapsing을 방지하기 위해 **stop-gradient** 기법을 사용한다.

Self-supervised learning은 학생이 교과서를 보며 스스로 공부하는 것과 비슷하다. 학생이 스스로 공부할 때 여러 문제를 푸는 것처럼, self-supervised learning에서는 모델이 데이터를 여러 방식으로 변형하여 학습한다. 이 과정에서 모든 답을 같은 값으로 내버리는 collapsing 문제를 막기 위해 SimCLR와 MoCo는 학생이 잘못된 답을 피하도록 negative sample을 제공하는 방식이다. \
반면 BYOL은 이러한 negative sample 없이도 학습할 수 있는 방식을 제안하였다. SimSiam은 이러한 접근 방식을 더욱 단순화하여 negative pair 없이 positive pair 만을 사용하고, stop-gradient 기법을 통해 collapsing 문제를 해결한다.

![figure3](https://github.com/yyeongha/DeepCV_study/blob/main/week01/img/simsiam/figure3.png?raw=true)

## 2. Method

SimSiam의 구조는 다음과 같다:

1. 하나의 이미지 $x$를 두 가지 방식으로 변형하여 두 개의 augmented 이미지를 생성한다 ($x_1$, $x_2$).
2. 이 두 이미지는 같은 encoder 네트워크 $f$를 통과한다.
3. encoder를 통과한 두 벡터 중 하나만 predictor $h$를 통과한다:
   - $p_1 = h(f(x_1))$
   - $z_2 = f(x_2)$

![method](https://github.com/yyeongha/DeepCV_study/blob/main/week01/img/simsiam/method.png?raw=true)

### Loss

SimSiam은 두 벡터 $p_1$과 $z_2$의 negative cosine similarity를 최소화하는 loss를 사용한다:

![equation2](https://github.com/yyeongha/DeepCV_study/blob/main/week01/img/simsiam/equation2.png?raw=true)

이러한 비교를 두 벡터의 순서를 바꾸어 한 번 더 수행하여 symmetric loss를 사용한다:



이 방식은 두 명의 학생이 서로의 답을 교차 검사하는 것과 비슷하다. 한 학생이 문제를 풀고 다른 학생이 그 답을 검토한 후, 다시 원래 학생이 검토한 답을 평가하는 방식이다. 이를 통해 모델이 더 정확한 답을 찾도록 돕는다.

### Stop-gradient

SimSiam의 핵심적인 구조는 여기에 stop-gradient를 추가하는 것이다:

![equation](https://github.com/yyeongha/DeepCV_study/blob/main/week01/img/simsiam/equation.png?raw=true)

이를 통해 $x_2$에서 $z_2$로부터는 gradient를 전달받지 않고, $p_2$를 통해 gradient를 전달받게 된다. 이는 학생이 문제를 푼 후 답을 바로 고치지 않고, 잠시 멈추고 다른 방식으로 검토하도록 하는 것과 유사하다. 이렇게 하면 모델이 쉽게 잘못된 방향으로 치우치지 않고 학습할 수 있다.

### Pseudo-code

SimSiam의 알고리즘은 다음과 같다:

```python
# f: backbone + projection MLP
# h: prediction MLP

for x in loader: # load a minibatch x with n samples
    x1, x2 = aug(x), aug(x) # random augmentation
    z1, z2 = f(x1), f(x2) # projections, n-by-d
    p1, p2 = h(z1), h(z2) # predictions, n-by-d

    L = D(p1, z2)/2 + D(p2, z1)/2 # loss

    L.backward() # back-propagate
    update(f, h) # SGD update
```

## 3. Empirical study

SimSiam의 실험 결과는 다음과 같다. 다양한 설정에서 stop-gradient의 유무에 따른 성능 비교를 통해 SimSiam의 효용성을 입증하였다.

### Stop Gradient

SimSiam 모델에서 stop-gradient 유무에 따른 성능 비교 결과는 다음과 같다:

- stop-gradient를 사용하지 않으면 train loss가 -1로 수렴하며, 이는 모든 output이 constant vector로 고정되고, loss가 최소값인 -1로 고정됨을 의미한다.
- stop-gradient를 사용한 경우, output std value가 \(1/\sqrt{d}\)로 수렴하며, 이는 collapsing이 일어나지 않고 output이 unit hypersphere에 분포됨을 의미한다.
- Linear classification을 시도했을 때, stop-gradient를 적용한 SimSiam은 67.7%의 accuracy를 보이며, stop-gradient가 없는 경우 accuracy는 0.1%로 전혀 학습되지 않음을 보여준다.

이는 학생이 문제를 풀 때, 즉각적인 피드백 없이 스스로 답을 찾아가는 과정을 거치면 더 나은 학습 결과를 얻을 수 있음을 보여준다.

### Predictor

- predictor가 없는 경우에는 collapsing이 발생한다.
- predictor를 random weight로 고정한 경우에는 loss가 수렴하지 않는다.
- predictor의 learning rate를 decay하지 않는 경우, accuracy가 +0.4% 상승한다.

### Batch Size

Batch size와는 무관한 성능을 보인다. 이는 학생이 문제를 푸는 과정에서 문제의 양이 크게 중요하지 않음을 시사한다.

### Batch Normalization

Batch Normalization(BN)을 적절한 위치에 사용하면 성능 향상은 있지만, collapsing 방지와는 무관함을 보였다.

### Similarity Function

cosine similarity를 cross-entropy similarity로 대체했을 때도 collapsing 없이 준수한 성능을 보였다. 이는 다양한 평가 기준을 사용해도 SimSiam이 안정적으로 학습할 수 있음을 의미한다.

### Symmetrization

asymmetric loss에서도 collapsing 현상은 발견되지 않았다. 이는 학습 과정에서 일부 비대칭적인 요소가 있어도 SimSiam이 잘 동작할 수 있음을 보여준다.

요약하면, stop-gradient를 적용하지 않거나 predictor MLP를 제거하는 경우에만 collapsing이 나타나며, batch size, BN, similarity function, symmetric loss는 collapsing을 방지하는 주요 요소가 아니다.

## 4. Hypothesis

SimSiam의 작동원리를 Expectation Maximization(EM)과 비슷하게 해석할 수 있다:

\[ L(\theta, \eta) = E_{x, T}[\|\mathcal{F}_\theta(T(x)) - \eta_x\|^2_2] \]

여기서 $\mathcal{F}$는 네트워크, $T$는 augmentation을 의미한다. 이를 alternating algorithm으로 해결할 수 있으며, $\theta$와 $\eta$를 각각 고정하면서 최적화할 수 있다.

이 과정은 학생이 여러 번의 검토와 수정을 통해 최적의 답을 찾아가는 과정과 유사하다. 먼저 $\theta$를 고정한 상태에서 $\eta$를 최적화하고, 이후 $\eta$를 고정한 상태에서 $\theta$를 최적화하는 방식이다.

## 5. Comparison
![table4](https://github.com/yyeongha/DeepCV_study/blob/main/week01/img/simsiam/table4.png?raw=true)

SimSiam은 SimCLR, MoCo, BYOL, SwAV와 비교하여 경쟁력 있는 결과를 보이며, 특히 100-epoch pre-training에서 가장 높은 accuracy를 달성했다. Transfer Learning에서도 SimSiam의 representation은 다른 태스크에서도 높은 성능을 보였다.

이와 같은 결과는 SimSiam의 단순한 구조가 다양한 학습 상황에서도 높은 성능을 유지할 수 있음을 보여준다. 이는 학생이 문제를 풀 때 복잡한 전략보다는 기본 원리에 충실한 학습이 효과적일 수 있음을 시사한다.

## 6. Conclusion

SimSiam은 간단한 설계에도 불구하고 강력한 성능을 보여주며, Siamese 네트워크의 역할을 재고하게 만든다. 이 연구는 unsupervised representation learning에서 Siamese 네트워크의 중요성을 강조하며, 향후 연구 방향에 중요한 시사점을 제공한다.

이를 통해 self-supervised learning에서 복잡한 전략보다 기본 원리에 충실한 접근이 더 효과적일 수 있음을 알 수 있다. 앞으로의 연구는 SimSiam의 간단한 구조를 바탕으로 더욱 효율적이고 효과적인 학습 방법을 개발하는 데 기여할 것이다.