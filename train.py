import os
import sys
import argparse
import logging
from torch.utils.tensorboard import SummaryWriter
from net import BottleNeck,EAMNet
import torch
from torchvision import transforms
import torch.optim as optim
from dataset import MyDataset
from tqdm import tqdm
import math
import random
import warnings
import json
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler as lr_scheduler
from data_split2 import data_set_split

parser = argparse.ArgumentParser(description='Training with pytorch')
parser.add_argument("--dataset_type",default='My_data',type=str,help='type of dataset')
parser.add_argument("--root_datasets",default=r"/root/autodl-nas/raw_image",help='Dataset directory path')
parser.add_argument("--target_data_folder",default=r"/root/autodl-nas/prosessed_image",help='Dataset directory path')

parser.add_argument('--balance_data', action='store_true',help="Balance training data by down-sampling more frequent labels.")
parser.add_argument('--net', default="EAMNet")
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, # 5e-4
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--resume', default=None, type=str, help='Checkpoint state_dict file to resume training from')
# Train params
parser.add_argument('--batch_size', default=16, type=int,
                    help='Batch size for training')
parser.add_argument('--num_epochs', default=200, type=int,
                    help='the number epochs')
parser.add_argument('--num_workers', default=8, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--use_cuda', default=True, type=bool,
                    help='Use CUDA to train model')
parser.add_argument('--checkpoint_folder', default='1_model_newspaper_hecheng_ToDesk/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--lrf', type=float, default=0.01)
parser.add_argument('--seed', default=3, type=int,help='seed for initializing training. ')

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')



# 参数分别是 训练loader, net, criterion(loss),optimizer,device,debug_steps,epoch
def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    mean_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()

    data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        images, labels = data

        pred = model(images.to(device))

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses

        data_loader.desc = "[epoch {}] mean loss {}".format(epoch, round(mean_loss.item(), 3))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return mean_loss.item()

# @torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()

    # 验证样本总个数
    total_num = len(data_loader.dataset)

    # 用于存储预测正确的样本个数
    sum_num = torch.zeros(1).to(device)

    data_loader = tqdm(data_loader, file=sys.stdout)
    with torch.no_grad():   # 可以跟上面的@torch.no_grad()  替换掉
        for step, data in enumerate(data_loader):
            images, labels = data
            pred = model(images.to(device))
            pred = torch.max(pred, dim=1)[1]
            sum_num += torch.eq(pred, labels.to(device)).sum()

    return sum_num.item() / total_num


def adjust_learning_rate(args):
    lf = lambda x: ((1 + math.cos(x * math.pi / args.num_epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    return lf

def class2json(train_dataset):
    flower_list = train_dataset.image_cla
    cla_dict = dict((val, key) for key, val in flower_list.items())
    json_str = json.dumps(cla_dict, indent=9)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

def main():
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")
    
    if torch.cuda.is_available() and args.use_cuda:
        torch.backends.cudnn.benchmark = True
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

        
        logging.info("Use Cuda.")
    logging.info(args) 
    tb_writer = SummaryWriter()
    if args.net == 'U_SeResnet':
        create_net = EAMNet(block=BottleNeck,layers=[3,4,6,3],num_classes=3).to(device)

    transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            transforms.Normalize([0.20132038, 0.20132038, 0.20132038], [0.24066962, 0.24066962, 0.24066962])
        ]),
        "val": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            transforms.Normalize([0.20132038, 0.20132038, 0.20132038], [0.24066962, 0.24066962, 0.24066962])
        ])}
    train_images_path, train_images_labels, val_images_path, val_images_labels = data_set_split(args.root_datasets,args.target_data_folder)
    train_dataset = MyDataset(image_path=train_images_path, image_cla=train_images_labels, transform=transform["train"])
    val_dataset = MyDataset(image_path=val_images_path, image_cla=val_images_labels, transform=transform["val"])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=args.num_workers,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=args.num_workers,
                                             collate_fn=val_dataset.collate_fn)
    pg = [p for p in create_net.parameters() if p.requires_grad]
    '''
    调整学习率：是基于在每一个epoch调整，同时学习率的调整要在优化器参数之后更新。
    "即先优化器更新，在学习率更新"
    '''
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=1E-4)

    lf = adjust_learning_rate(args)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    max_acc=0
    for epoch in range(args.num_epochs):
        # train
        mean_loss = train_one_epoch(model=create_net,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    device=device,
                                    epoch=epoch)

        scheduler.step()

        # validate
        acc = evaluate(model=create_net,
                       data_loader=val_loader,
                       device=device)
        logging.info("[epoch {}] accuracy: {}".format(epoch, round(acc, 3)))
        tags = ["loss", "accuracy", "learning_rate"]
        tb_writer.add_scalar(tags[0], mean_loss, epoch)
        tb_writer.add_scalar(tags[1], acc, epoch)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)
        if acc>max_acc:
            max_acc=acc
            print(f"save best model,第{epoch}轮")
            torch.save(create_net.state_dict(), r"/root/autodl-tmp/weights/model_new-{}-of-{}-{}-{}.pth".format(epoch,args.num_epochs,mean_loss,acc))

if __name__ == '__main__':
    main()
