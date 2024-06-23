import torch
import torch.nn as nn


class SimSiam(nn.Module):
    """
    SimSiam 모델을 구성
    """
    def __init__(self, base_encoder, dim=2048, pred_dim=512):
        """
        dim: 특징 차원 (기본값: 2048)
        pred_dim: 예측기의 은닉 차원 (기본값: 512)
        """
        super(SimSiam, self).__init__()

        # 인코더 생성
        # num_classes는 출력 fc 차원이고, 마지막 BN을 0으로 초기화
        self.encoder = base_encoder(num_classes=dim, zero_init_residual=True)

        # 3-레이어 프로젝터 구축
        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # 첫 번째 레이어
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # 두 번째 레이어
                                        self.encoder.fc,
                                        nn.BatchNorm1d(dim, affine=False)) # 출력 레이어
        self.encoder.fc[6].bias.requires_grad = False # 해킹: BN 뒤에 bias를 사용하지 않음

        # 2-레이어 예측기 구축
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # 은닉 레이어
                                        nn.Linear(pred_dim, dim)) # 출력 레이어
    # 순전파 함수
    def forward(self, x1, x2):
        """
        입력:
            x1: 첫 번째 이미지 뷰
            x2: 두 번째 이미지 뷰
            x1, x2는 두개의 증강된 이미지 뷰
        출력:
            p1, p2, z1, z2: 네트워크의 예측기와 타겟
            자세한 표기법은 https://arxiv.org/abs/2011.10566 논문의 섹션 3 참조
        """

        # 하나의 뷰에 대한 특징을 계산합니다.
        z1 = self.encoder(x1) # NxC
        z2 = self.encoder(x2) # NxC

        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC

        # 예측기와 타겟 반환, 논문 figure 1의 표기법을 따름
        return p1, p2, z1.detach(), z2.detach() # detach()를 사용하여 z1과 z2에서 그래디언트를 분리(stop-gradient)
    
