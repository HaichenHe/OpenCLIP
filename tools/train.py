import os
import sys
import time
import argparse

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler
import torchvision
import numpy as np

from utils.utils import init_distributed_mode, epoch_saving, best_saving, AverageMeter, reduce_tensor, accuracy, create_logits, gen_label, gather_labels
from utils.logger import setup_logger

from pathlib import Path
import yaml
from dotmap import DotMap

import datetime
import shutil
from contextlib import suppress

from utils.solver import _optimizer, _lr_scheduler
from utils.data_loader import train_data_loader, val_data_loader
from slowfast.models.text_prompt import text_prompt

from slowfast.utils.misc import log_model_info
from slowfast.models import build_model
import slowfast.models.losses as losses



class AllGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor):
        output = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
        torch.distributed.all_gather(output, tensor)
        ctx.rank = dist.get_rank()
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank : ctx.batch_size * (ctx.rank + 1)],
            None,
        )

allgather = AllGather.apply

def update_dict(dict):
    new_dict = {}
    for k, v in dict.items():
        new_dict[k.replace('module.', '')] = v
    return new_dict

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', type=str, default='clip.yaml', help='global config file')
    parser.add_argument('--log_time', default='001')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')                        
    parser.add_argument("--local-rank", type=int,
                        help='local rank for DistributedDataParallel')
    parser.add_argument(
        "--precision",
        choices=["amp", "fp16", "fp32"],
        default="amp",
        help="Floating point precition."
    )                        
    args = parser.parse_args()
    return args



def main(args):
    global best_prec1
    """ Training Program """
    init_distributed_mode(args)
    if args.distributed:
        print('[INFO] turn on distributed train', flush=True)
    else:
        print('[INFO] turn off distributed train', flush=True)

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = DotMap(config)

    # fix the seed for reproducibility
    seed = config.seed + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        cudnn.benchmark = True

    model = build_model(config)

    for name, param in model.module.model.named_parameters():
        param.requires_grad_(False) 

    logger_dir = os.path.join('exps', config['data']['dataset'], config['network']['arch'], args.log_time)
    if dist.get_rank() == 0:
        Path(logger_dir).mkdir(parents=True, exist_ok=True)
        shutil.copy(args.config, logger_dir)
        shutil.copy('train.py', logger_dir)
    # build logger, print env and config
    logger = setup_logger(output=logger_dir,
                          distributed_rank=dist.get_rank(),
                          name=f'OpenCLIP')
    log_model_info(model, config,logger, use_train_input=False)

    train_loader, train_data = train_data_loader(config, logger)
    val_loader = val_data_loader(config, logger)
    
    criterion = losses.get_loss_func(config.solver.loss_type)(reduction="mean")

    start_epoch = config.solver.start_epoch
    
    if config.pretrain:
        if os.path.isfile(config.pretrain):
            logger.info("=> loading checkpoint '{}'".format(config.pretrain))
            checkpoint = torch.load(config.pretrain, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            del checkpoint
        else:
            logger.info("=> no checkpoint found at '{}'".format(config.resume))
    
    if config.resume:
        if os.path.isfile(config.resume):
            logger.info("=> loading checkpoint '{}'".format(config.resume))
            checkpoint = torch.load(config.resume, map_location='cpu')
            model.load_state_dict(update_dict(checkpoint['model_state_dict']))
            start_epoch = checkpoint['epoch'] + 1
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                   .format(config.evaluate, checkpoint['epoch']))
            del checkpoint
        else:
            logger.info("=> no checkpoint found at '{}'".format(config.pretrain))


    optimizer = _optimizer(config, model)
    lr_scheduler = _lr_scheduler(config, optimizer)
    scaler = GradScaler() if args.precision == "amp" else None

    best_prec1 = 0.0
    classes = text_prompt(train_data)
    n_class = classes.size(0)
    if config.solver.evaluate:
        logger.info(("===========evaluate==========="))
        prec1, output_list, labels_list = validate(
            start_epoch,
            val_loader, classes, device,
            model, config, n_class, logger)
        return



    for epoch in range(start_epoch, config.solver.epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)        

        train(model, train_loader, optimizer, criterion, scaler,
              epoch, device, lr_scheduler, config, classes, logger)

        if (epoch+1) % config.logging.eval_freq == 0:
            prec1, output_list, labels_list = validate(epoch, val_loader, classes, device, model, config, n_class, logger)

            if dist.get_rank() == 0:
                is_best = prec1 > best_prec1
                best_prec1 = max(prec1, best_prec1)
                logger.info('Testing: {}/{}'.format(prec1,best_prec1))
                logger.info('Saving:')
                filename = "{}/last_model.pt".format(logger_dir)

                epoch_saving(epoch, model.module, optimizer, filename)
                if is_best:
                    best_saving(logger_dir, epoch, model.module, optimizer)
                    


def train(model, train_loader, optimizer, criterion, scaler,
          epoch, device, lr_scheduler, config, classes, logger):
    """ train a epoch """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    img_losses = AverageMeter()
    text_losses = AverageMeter()

    model.train()
    autocast = torch.cuda.amp.autocast if args.precision == 'amp' else suppress
    end = time.time()
    for i,(images, list_id) in enumerate(train_loader):
        if config.solver.type != 'monitor':
            if (i + 1) == 1 or (i + 1) % 10 == 0:
                lr_scheduler.step(epoch + i / len(train_loader))
        # lr_scheduler.step()

        data_time.update(time.time() - end)
        # b t3 h w
        images = images.view((-1,config.data.num_segments,3)+images.size()[-2:])  # bt 3 h w
        b,t,c,h,w = images.size()
        images= images.view(-1,c,h,w)
        texts = classes # n_cls 77

        with autocast():
            texts = texts[list_id]  # bs 77
            image_embedding, cls_embedding = model(images, texts)
            image_embedding = image_embedding.view(b,t,-1) #b,t,d
            image_embedding = image_embedding.mean(1) # b,d
            image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
            cls_embedding = cls_embedding / cls_embedding.norm(dim=-1, keepdim=True)
            # gather
            image_embedding = allgather(image_embedding)
            cls_embedding = allgather(cls_embedding)    

            logits = image_embedding @ cls_embedding.t() #b,b
            logits = model.module.model.logit_scale.exp() * logits
                
            list_id = gather_labels(list_id.to(device))  # bs -> n_gpu * bs
            ground_truth = torch.tensor(gen_label(list_id),dtype=image_embedding.dtype,device=device)
            # gt = [bs bs]
            loss_imgs = criterion(logits, ground_truth)
            loss_texts = criterion(logits.T, ground_truth)
            loss = (loss_imgs + loss_texts)/2
        
            # loss regularization
            loss = loss / config.solver.grad_accumulation_steps

        if scaler is not None:
            # back propagation
            scaler.scale(loss).backward()
            if (i + 1) % config.solver.grad_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()  # reset gradient
        else:
            # back propagation
            loss.backward()
            if (i + 1) % config.solver.grad_accumulation_steps == 0:
                optimizer.step()  # update param
                optimizer.zero_grad()  # reset gradient

        losses.update(loss.item(), logits.size(0))


        batch_time.update(time.time() - end)
        end = time.time()
        cur_iter = epoch * len(train_loader) + i
        max_iter = config.solver.epochs * len(train_loader)
        eta_sec = batch_time.avg * (max_iter - cur_iter + 1)
        eta_sec = str(datetime.timedelta(seconds=int(eta_sec)))

        if i % config.logging.print_freq == 0:
            logger.info(('Epoch: [{0}][{1}/{2}], lr: {lr:.2e}, eta: {3}\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                             epoch, i, len(train_loader), eta_sec, batch_time=batch_time, data_time=data_time, loss=losses,
                             lr=optimizer.param_groups[-1]['lr'])))




def validate(epoch, val_loader, classes, device, model, config, n_class, logger):
    top1 = AverageMeter()
    top5 = AverageMeter()
    sims_list = []
    labels_list = []
    model.eval()

    with torch.no_grad():
        text_inputs = classes.to(device)  # [n_cls, 77]
        cls_feature = model.module.encode_text(text_inputs)  # [n_cls, feat_dim]
        for i, (image, class_id) in enumerate(val_loader):
            image = image.view((-1, config.data.num_segments, 3) + image.size()[-2:])
            b, t, c, h, w = image.size()
            class_id = class_id.to(device)
            image_input = image.to(device).view(-1, c, h, w)
            image_features = model.module.encode_image(image_input).view(b, t, -1) #b,t,d
            image_features = image_features.mean(1) #b,d
            similarity = image_features @ cls_feature.t() # b,n_cls
            similarity = similarity.softmax(dim=-1)  

            prec = accuracy(similarity, class_id, topk=(1, 5))
            prec1 = reduce_tensor(prec[0])
            prec5 = reduce_tensor(prec[1])

            top1.update(prec1.item(), class_id.size(0))
            top5.update(prec5.item(), class_id.size(0))

            if i % config.logging.print_freq == 0:
                logger.info(
                    ('Test: [{0}/{1}]\t'
                     'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                     'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                         i, len(val_loader), top1=top1, top5=top5)))
    logger.info(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
        .format(top1=top1, top5=top5)))
    
    return top1.avg, None, None



if __name__ == '__main__':
    args = get_parser() 
    main(args)

