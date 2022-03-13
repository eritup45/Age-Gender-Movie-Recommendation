import argparse
import better_exceptions
from pathlib import Path
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.optim.lr_scheduler import StepLR
import torch.utils.data
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import pretrainedmodels
import pretrainedmodels.utils
from model import get_model
from dataset import FaceDataset
from defaults import _C as cfg

reduce_gender_loss_weight=0.1   # TODO: change 2 parameters form (parser)

def get_args():
    model_names = sorted(name for name in pretrainedmodels.__dict__
                         if not name.startswith("__")
                         and name.islower()
                         and callable(pretrainedmodels.__dict__[name]))
    parser = argparse.ArgumentParser(description=f"available models: {model_names}",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_dir", type=str, required=True, help="Data root directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint if any")
    parser.add_argument("--checkpoint", type=str, default="checkpoint", help="Checkpoint directory")
    parser.add_argument("--tensorboard", type=str, default=None, help="Tensorboard log directory")
    parser.add_argument('--multi_gpu', action="store_true", help="Use multi GPUs (data parallel)")
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")
    args = parser.parse_args()
    return args


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


def train(train_loader, model, criterion, optimizer, epoch, device):
    model.train()
    age_loss_monitor = AverageMeter()
    age_accuracy_monitor = AverageMeter()
    gender_accuracy_monitor = AverageMeter()    # Add to monitor

    # TODO: change x->img, y->age
    with tqdm(train_loader) as _tqdm:
        for x, y, y_gender in _tqdm:
            x = x.to(device)
            y = y.to(device)
            y_gender = y_gender.to(device)

            # compute output
            age_out, gender_out = model(x)

            # calc loss
            age_loss = criterion(age_out, y)
            gender_loss = criterion(gender_out, y_gender)

            # *Note: reduce loss on gender so the model focus on age pred
            gender_loss *= reduce_gender_loss_weight

            loss = age_loss + gender_loss

            cur_loss = age_loss.item()  # TODO: cur_loss = age_loss.item() + gender_loss.item()

            # calc accuracy
            _, age_pred = age_out.max(1)
            _, gender_pred = gender_out.max(1)
            age_correct_num = age_pred.eq(y).sum().item()
            gender_correct_num = gender_pred.eq(y_gender).sum().item()

            # measure accuracy and record loss
            sample_num = x.size(0)
            age_loss_monitor.update(cur_loss, sample_num)
            age_accuracy_monitor.update(age_correct_num, sample_num)
            gender_accuracy_monitor.update(gender_correct_num, sample_num)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _tqdm.set_postfix(OrderedDict(stage="train", epoch=epoch, age_loss=age_loss_monitor.avg),
                            age_acc=age_accuracy_monitor.avg, age_correct=age_correct_num, 
                            gender_acc=gender_accuracy_monitor.avg, sample_num=sample_num)

    return age_loss_monitor.avg, age_accuracy_monitor.avg


def validate(validate_loader, model, criterion, epoch, device):
    model.eval()
    age_loss_monitor = AverageMeter()
    gender_accuracy_monitor = AverageMeter()    # Add to monitor
    age_accuracy_monitor = AverageMeter()
    age_preds = []
    gt = []

    with torch.no_grad():
        with tqdm(validate_loader) as _tqdm:
            for i, (x, y, y_gender) in enumerate(_tqdm):
                x = x.to(device)
                y = y.to(device)
                y_gender = y_gender.to(device)

                # compute output
                age_out, gender_out = model(x)
                # Print Result
                # print(f'age_out:{age_out.max(1)[1]}, y_age:{y}, gender_out:{gender_out.max(1)[1]}, y_gender:{y_gender}')
                age_preds.append(F.softmax(age_out, dim=-1).cpu().numpy())
                gt.append(y.cpu().numpy())

                # calc loss
                age_loss = criterion(age_out, y)
                gender_loss = criterion(gender_out, y_gender)
                cur_loss = age_loss.item()  # TODO: cur_loss = age_loss.item() + gender_loss.item()
                
                # calc accuracy
                _, age_pred = age_out.max(1)
                _, gender_pred = gender_out.max(1)
                age_correct_num = age_pred.eq(y).sum().item()
                gender_correct_num = gender_pred.eq(y_gender).sum().item()

                # measure accuracy and record loss
                sample_num = x.size(0)
                age_loss_monitor.update(cur_loss, sample_num)
                age_accuracy_monitor.update(age_correct_num, sample_num)
                gender_accuracy_monitor.update(gender_correct_num, sample_num)
                # _tqdm.set_postfix(OrderedDict(stage="val", epoch=epoch, age_loss=age_loss_monitor.avg),
                #                   age_acc=age_accuracy_monitor.avg, age_correct=age_correct_num, sample_num=sample_num)
                _tqdm.set_postfix(OrderedDict(stage="val", epoch=epoch, age_loss=age_loss_monitor.avg),
                        age_acc=age_accuracy_monitor.avg, age_correct=age_correct_num, 
                        gender_acc=gender_accuracy_monitor.avg, sample_num=sample_num)

    age_preds = np.concatenate(age_preds, axis=0)
    gt = np.concatenate(gt, axis=0)
    ages = np.arange(0, 101)
    ave_preds = (age_preds * ages).sum(axis=-1)
    diff = ave_preds - gt
    mae = np.abs(diff).mean()

    return age_loss_monitor.avg, age_accuracy_monitor.avg, mae, gender_accuracy_monitor.avg


def main():
    args = get_args()

    if args.opts:
        cfg.merge_from_list(args.opts)

    cfg.freeze()
    start_epoch = 0
    checkpoint_dir = Path(args.checkpoint)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # create model
    print("=> creating model '{}'".format(cfg.MODEL.ARCH))
    model = get_model(model_name=cfg.MODEL.ARCH)

    if cfg.TRAIN.OPT == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.TRAIN.LR,
                                    momentum=cfg.TRAIN.MOMENTUM,
                                    weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    model = model.to(device)

    # # TODO: Add arguments
    # if torch.cuda.device_count() > 1:
    #   print("Let's use [1,2,4,5] GPUs!")
    #   # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    #   model = nn.DataParallel(model,device_ids=[1,2,4,5])
    # model.to(device)

    # optionally resume from a checkpoint
    resume_path = args.resume

    if resume_path:
        if Path(resume_path).is_file():
            print("=> loading checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path, map_location="cpu")
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_path, checkpoint['epoch']))
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(resume_path))

    if args.multi_gpu:
        model = nn.DataParallel(model)

    if device == "cuda":
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss().to(device)
    train_dataset = FaceDataset(args.data_dir, "train", img_size=cfg.MODEL.IMG_SIZE, augment=True,
                                age_stddev=cfg.TRAIN.AGE_STDDEV)
    train_loader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True,
                              num_workers=cfg.TRAIN.WORKERS, drop_last=True)

    val_dataset = FaceDataset(args.data_dir, "valid", img_size=cfg.MODEL.IMG_SIZE, augment=False)
    val_loader = DataLoader(val_dataset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False,
                            num_workers=cfg.TRAIN.WORKERS, drop_last=False)    
    
    scheduler = StepLR(optimizer, step_size=cfg.TRAIN.LR_DECAY_STEP, gamma=cfg.TRAIN.LR_DECAY_RATE,
                       last_epoch=start_epoch - 1)
    best_val_mae = 10000.0
    train_writer = None

    if args.tensorboard is not None:
        opts_prefix = "_".join(args.opts)
        train_writer = SummaryWriter(log_dir=args.tensorboard + "/" + opts_prefix + "_train")
        val_writer = SummaryWriter(log_dir=args.tensorboard + "/" + opts_prefix + "_val")

    for epoch in range(start_epoch, cfg.TRAIN.EPOCHS):
        # train
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, device)

        # validate
        val_loss, val_acc, val_mae, val_gen_acc = validate(val_loader, model, criterion, epoch, device)

        if args.tensorboard is not None:
            train_writer.add_scalar("age_loss", train_loss, epoch)
            train_writer.add_scalar("age_acc", train_acc, epoch)
            val_writer.add_scalar("age_loss", val_loss, epoch)
            val_writer.add_scalar("age_acc", val_acc, epoch)
            val_writer.add_scalar("age_mae", val_mae, epoch)

        # checkpoint
        if val_mae < best_val_mae:
            print(f"=> [epoch {epoch:03d}] best val mae was improved from {best_val_mae:.3f} to {val_mae:.3f}")
            model_state_dict = model.module.state_dict() if args.multi_gpu else model.state_dict()
            torch.save(
                {
                    'epoch': epoch + 1,
                    'arch': cfg.MODEL.ARCH,
                    'state_dict': model_state_dict,
                    'optimizer_state_dict': optimizer.state_dict()
                },
                str(checkpoint_dir.joinpath("epoch{:03d}_{:.5f}_{:.4f}.pth".format(epoch, val_loss, val_mae)))
            )
            best_val_mae = val_mae
        else:
            print(f"=> [epoch {epoch:03d}] best val mae was not improved from {best_val_mae:.3f} ({val_mae:.3f})")

        # adjust learning rate
        scheduler.step()

    print("=> training finished")
    print(f"additional opts: {args.opts}")
    print(f"best val mae: {best_val_mae:.3f}")


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()
