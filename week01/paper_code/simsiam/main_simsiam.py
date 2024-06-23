import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import simsiam.loader
import simsiam.builder

# 모델 이름을 가져와 정렬
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

# 명령줄 인자 파서를 설정.
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='데이터셋 경로')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='모델 아키텍처: ' +
                        ' | '.join(model_names) +
                        ' (기본값: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='데이터 로딩 작업자 수 (기본값: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='전체 에포크 수 (기본값: 100)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='수동 에포크 번호 (재시작 시 유용)')
parser.add_argument('-b', '--batch-size', default=512, type=int,
                    metavar='N',
                    help='미니 배치 크기 (기본값: 512), 이는 데이터 병렬 또는 분산 데이터 병렬 사용 시 현재 노드의 모든 GPU의 총 배치 크기')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='초기 (기본) 학습률', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='SGD 솔버의 모멘텀')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='가중치 감쇠 (기본값: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='출력 빈도 (기본값: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='최신 체크포인트 경로 (기본값: 없음)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='분산 학습을 위한 노드 수')
parser.add_argument('--rank', default=-1, type=int,
                    help='분산 학습을 위한 노드 랭크')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='분산 학습 설정을 위한 URL')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='분산 백엔드')
parser.add_argument('--seed', default=None, type=int,
                    help='훈련 초기화를 위한 시드')
parser.add_argument('--gpu', default=None, type=int,
                    help='사용할 GPU ID')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='멀티 프로세싱 분산 학습을 사용하여 각 노드에 N개의 프로세스를 실행함 이는 단일 노드 또는 다중 노드 데이터 병렬 학습을 위한 가장 빠른 방법임')

# SimSiam 관련 추가 설정
parser.add_argument('--dim', default=2048, type=int,
                    help='특징 차원 (기본값: 2048)')
parser.add_argument('--pred-dim', default=512, type=int,
                    help='예측기의 은닉 차원 (기본값: 512)')
parser.add_argument('--fix-pred-lr', action='store_true',
                    help='예측기의 학습률 고정')

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('훈련에 시드를 설정했음 '
                      '이로 인해 CUDNN 결정론적 설정이 켜져 훈련이 상당히 느려질 수 있다 '
                      '체크포인트에서 재시작할 때 예상치 못한 동작이 발생할 수 있다.')

    if args.gpu is not None:
        warnings.warn('특정 GPU를 선택, 이는 데이터 병렬 처리를 완전히 비활성화하는 것을 의미할 수 있다')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # 각 노드에 ngpus_per_node 프로세스를 가지므로 총 world_size를 조정해야 함
        args.world_size = ngpus_per_node * args.world_size
        # torch.multiprocessing.spawn을 사용하여 분산 프로세스를 실행: main_worker 함수 실행
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # 단순히 main_worker 함수 호출
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # 마스터가 아닌 경우 출력 억제
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("훈련에 GPU: {} 사용".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # 멀티 프로세싱 분산 학습의 경우, rank는 모든 프로세스 중 글로벌 rank여야 함.
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()

    # 모델 생성
    print("=> 모델 생성 '{}'".format(args.arch))
    model = simsiam.builder.SimSiam(
        models.__dict__[args.arch],
        args.dim, args.pred_dim)

    # 배치 크기 변경 전 학습률 추론
    init_lr = args.lr * args.batch_size / 256

    if args.distributed:
        # SyncBatchNorm 적용
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # 멀티 프로세싱 분산 학습의 경우, DistributedDataParallel 생성자는 항상 단일 장치 범위를 설정해야 함
        # 그렇지 않으면, DistributedDataParallel은 사용 가능한 모든 장치를 사용함
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # 각 프로세스 당 단일 GPU를 사용할 때, 배치 크기를 모든 GPU에 맞게 나눠야 함
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # device_ids가 설정되지 않은 경우, DistributedDataParallel은 모든 사용 가능한 GPU에 배치 크기를 나누고 할당.
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # 디버깅을 위해 다음 줄을 주석 처리함
        raise NotImplementedError("DistributedDataParallel만 지원됨")
    else:
        # 이 코드의 AllGather 구현(batch shuffle, queue update 등)은 DistributedDataParallel만 지원
        raise NotImplementedError("DistributedDataParallel만 지원됨")
    print(model) # SyncBatchNorm 이후 모델 출력

    # 손실 함수(기준) 및 옵티마이저 정의
    # 손실 함수로 코사인 유사도를 사용하여 두 개의 증강된 이미지 뷰 간의 유사성을 최대화함. 이는 논문에서 언급된 손실 함수와 일치
    criterion = nn.CosineSimilarity(dim=1).cuda(args.gpu)

    if args.fix_pred_lr:
        optim_params = [{'params': model.module.encoder.parameters(), 'fix_lr': False},
                        {'params': model.module.predictor.parameters(), 'fix_lr': True}]
    else:
        optim_params = model.parameters()

    optimizer = torch.optim.SGD(optim_params, init_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # 체크포인트에서 선택적으로 재개
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> 체크포인트 '{}' 로드 중".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # 단일 GPU에 맵핑
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> 체크포인트 '{}' 로드 완료 (에포크 {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> '{}' 에서 체크포인트를 찾을 수 없음".format(args.resume))

    cudnn.benchmark = True

    # 데이터 로딩 코드
    traindir = os.path.join(args.data, 'train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # MoCo v2의 증강: SimCLR과 유사 https://arxiv.org/abs/2002.05709
    augmentation = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # 강화되지 않음
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([simsiam.loader.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    train_dataset = datasets.ImageFolder(
        traindir,
        simsiam.loader.TwoCropsTransform(transforms.Compose(augmentation)))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, init_lr, epoch, args)

        # 한 에포크 동안 모델 훈련
        train(train_loader, model, criterion, optimizer, epoch, args)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=False, filename='checkpoint_{:04d}.pth.tar'.format(epoch))

# 두 개의 증강된 이미지 뷰를 사용하여 모델의 예측 결과(p1, p2)와 타겟(z1, z2)를 비교하여 손실을 계산
# 계산된 손실을 역전파하여 모델 파라미터를 업데이트
def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # 훈련 모드로 전환
    model.train()

    end = time.time()
    for i, (images, _) in enumerate(train_loader):
        # 데이터 로딩 시간 측정
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)

        # 출력 및 손실 계산
        p1, p2, z1, z2 = model(x1=images[0], x2=images[1])
        loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5

        losses.update(loss.item(), images[0].size(0))

        # 그라디언트 계산 및 SGD 단계 수행
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 경과 시간 측정
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """평균 및 현재 값을 계산하고 저장"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

# 학습률 조정은 코사인 스케줄링을 사용하여 점진적으로 감소시킴
def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """스케줄에 따라 학습률을 감소시킴"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr


if __name__ == '__main__':
    main()
