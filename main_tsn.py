# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import os
import time
import shutil
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_

from ops_tsntsmgst.dataset import TSNDataSet
from ops_tsntsmgst.models_tsn import VideoNet
from ops_tsntsmgst.transforms import *
from opts import parser
from ops_tsntsmgst import dataset_config
from ops_tsntsmgst.utils import AverageMeter, accuracy
# from ops.temporal_shift import make_temporal_pool

from tensorboardX import SummaryWriter

best_prec1 = 0
best_prec1_test = 0
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

def main():
    global args, best_prec1, best_prec1_test
    args = parser.parse_args()

    num_class, args.train_list, args.val_list, args.root_path, prefix = dataset_config.return_dataset(args.dataset,
                                                                                                      args.modality)
    # ---args---
    if args.element_filter:
        full_arch_name = 'TSN_{}_{}'.format(args.net, args.arch)
    else:
        full_arch_name = 'TSN_{}'.format(args.arch)
    args.store_name = '_'.join(
        [full_arch_name, args.dataset, args.modality, args.consensus_type, 'segment%d' % args.num_segments,
         'e{}'.format(args.epochs)])
    # if args.pretrain != 'imagenet':
    #     args.store_name += '_{}'.format(args.pretrain)
    if args.lr_type != 'step':
        args.store_name += '_{}'.format(args.lr_type)
    else:
        args.store_name += '_s%dp%d'%(args.lr_steps[0], args.lr_steps[1])
    if args.dense_sample:
        args.store_name += '_dense_val10'
    elif args.twice_sample:
        args.store_name += '_twice'
    if args.non_local:
        args.store_name += '_nl'
    if args.suffix is not None:
        args.store_name += '_{}'.format(args.suffix)
    if args.element_filter:
        args.store_name += '_ef%d'%args.cdiv
    if args.dropout > 0:
        args.store_name += '_drp%s'%(str(args.dropout)[2:])
    args.store_name += '_lr%s'%(str(args.lr)[2:])
    args.store_name += '_b%d'%(args.batch_size)
    if args.no_partialbn:
        args.store_name += '_npb'
    if args.ef_lr5:
        args.store_name += '_eflr5'
    if args.loop:
        args.store_name += '_loop'
    # time info
    year = time.localtime(time.time()).tm_year
    month = time.localtime(time.time()).tm_mon
    day = time.localtime(time.time()).tm_mday
    hour = time.localtime(time.time()).tm_hour
    timeInfo = '_%02d%02d%02dh%02d'%(day, month, year-2000, hour)
    args.store_name += timeInfo
    print('storing name: ' + args.store_name)
    # args.root_log += '_240320'
    # args.root_model += '_240320'

    assert args.modality == 'RGB'

    check_rootfolders()

    #-----flip------
    if args.dataset == 'somethingv1' or args.dataset == 'somethingv2':
        target_transforms = {86:87, 87:86, 93:94, 94:93, 166:167, 167:166}
    else:
        target_transforms = None

    # --- model init---
    model = VideoNet(num_class, args.num_segments, args.modality,
                backbone=args.arch, net=args.net,
                consensus_type=args.consensus_type,
                dropout=args.dropout,
                partial_bn=not args.no_partialbn,
                ef_lr5=args.ef_lr5,
                # fc_lr5=not (args.tune_from and args.dataset in args.tune_from),
                non_local=args.non_local,
                element_filter=args.element_filter,
                stage=args.stage,
                cdiv=args.cdiv,
                loop=args.loop,
                target_transforms=target_transforms)

    crop_size = model.scale_size if args.full_res else model.input_size # 224 or 256 (scale_size)
    crop_size_val2 = model.input_size
    scale_size = model.scale_size # 256
    input_mean = model.input_mean
    input_std = model.input_std
    policies = model.get_optim_policies()
    train_augmentation = model.get_augmentation()

    model = torch.nn.DataParallel(model).cuda()
    #device_ids=args.gpus

    optimizer = torch.optim.SGD(policies,
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.resume:
        # if args.temporal_pool:  # early temporal pool so that we can load the state_dict
        #     make_temporal_pool(model.module.base_model, args.num_segments)
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(("=> loaded checkpoint '{}' (epoch {})"
                   .format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    # if args.tune_from:
    #     print(("=> fine-tuning from '{}'".format(args.tune_from)))
    #     sd = torch.load(args.tune_from)
    #     sd = sd['state_dict']
    #     model_dict = model.state_dict()
    #     if 'TSM' in args.tune_from:
    #         replace_dict = []
    #         for k, v in sd.items():
    #             if k not in model_dict and k.replace('.net', '') in model_dict:
    #                 print('=> Load after remove .net: ', k)
    #                 replace_dict.append((k, k.replace('.net', '')))
    #         for k, v in model_dict.items():
    #             if k not in sd and k.replace('.net', '') in sd:
    #                 print('=> Load after adding .net: ', k)
    #                 replace_dict.append((k.replace('.net', ''), k))

    #         for k, k_new in replace_dict:
    #             sd[k_new] = sd.pop(k)
    #     keys1 = set(list(sd.keys()))
    #     keys2 = set(list(model_dict.keys()))
    #     set_diff = (keys1 - keys2) | (keys2 - keys1)
    #     print('#### Notice: keys that failed to load: {}'.format(set_diff))
    #     if args.dataset not in args.tune_from:  # new dataset
    #         print('=> New dataset, do not load fc weights')
    #         sd = {k: v for k, v in sd.items() if 'fc' not in k}
    #     # if args.modality == 'Flow' and 'Flow' not in args.tune_from:
    #     #     sd = {k: v for k, v in sd.items() if 'conv1.weight' not in k}
    #     model_dict.update(sd)
    #     model.load_state_dict(model_dict)

    # if args.temporal_pool and not args.resume:
    #     make_temporal_pool(model.module.base_model, args.num_segments)

    cudnn.benchmark = True

    # Data loading code
    if args.modality != 'RGBDiff':
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()

    if args.modality == 'RGB':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 5

    train_loader = torch.utils.data.DataLoader(
        TSNDataSet(args.root_path, args.train_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=prefix,
                   transform=torchvision.transforms.Compose([
                       GroupScale((240,320)),
                       train_augmentation,
                       Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                       normalize,
                   ]), dense_sample=args.dense_sample),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
        drop_last=True)  # prevent something not % n_GPU

    if args.dense_sample:
        val_batch_size = args.batch_size//2
    elif args.twice_sample:
        val_batch_size = int(args.batch_size*2)
    else:
        val_batch_size = args.batch_size*4
    val_batch_size = max(1, val_batch_size)
    
    val_loader = torch.utils.data.DataLoader(
        TSNDataSet(args.root_path, args.val_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=prefix,
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                       GroupScale((240,320)),
                       # GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size_val2),
                       Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                       normalize,
                   ]), dense_sample=args.dense_sample),
        batch_size=val_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    if args.loss_type == 'nll':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        raise ValueError("Unknown loss type")

    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

    if args.evaluate:
        validate(val_loader, model, criterion, 0)
        return

    log_training = open(os.path.join(args.root_log, args.store_name, 'log.csv'), 'w')
    with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
        f.write(str(args))
    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr_type, args.lr_steps)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, log_training, tf_writer)

        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs-1:
            prec1 = validate(val_loader, model, criterion, epoch, log_training, tf_writer)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            tf_writer.add_scalar('acc/val_top1_best', best_prec1, epoch)

            output_best = 'Best val Prec@1: %.3f\n' % (best_prec1)
            print(output_best)
            log_training.write(output_best + '\n')
            log_training.flush()

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_prec1': best_prec1,
            }, is_best)

def train(train_loader, model, criterion, optimizer, epoch, log, tf_writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if args.no_partialbn:
        model.module.partialBN(False)
    else:
        model.module.partialBN(True)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        loss.backward()

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)

        optimizer.step()
        optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr'] * 0.1))  # TODO
            print(output)
            log.write(output + '\n')
            log.flush()

    tf_writer.add_scalar('loss/train', losses.avg, epoch)
    tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)
    tf_writer.add_scalar('acc/train_top5', top5.avg, epoch)
    tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)


def validate(val_loader, model, criterion, epoch, log=None, tf_writer=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                output = ('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
                print(output)
                if log is not None:
                    log.write(output + '\n')
                    log.flush()

    output = ('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
              .format(top1=top1, top5=top5, loss=losses))
    print(output)
    if log is not None:
        log.write(output + '\n')
        log.flush()

    if tf_writer is not None:
        tf_writer.add_scalar('loss/valCenter', losses.avg, epoch)
        tf_writer.add_scalar('acc/valCenter_top1', top1.avg, epoch)
        tf_writer.add_scalar('acc/valCenter_top5', top5.avg, epoch)

    return top1.avg


def save_checkpoint(state, is_best):
    filename = '%s/%s/ckpt.pth.tar' % (args.root_model, args.store_name)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))


def adjust_learning_rate(optimizer, epoch, lr_type, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if lr_type == 'step':
        decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
        lr = args.lr * decay
        decay = args.weight_decay
    elif lr_type == 'cos':
        import math
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * epoch / args.epochs))
        decay = args.weight_decay
    else:
        raise NotImplementedError
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


def check_rootfolders():
    """Create log and model folder"""
    folders_util = [args.root_log, args.root_model,
                    os.path.join(args.root_log, args.store_name),
                    os.path.join(args.root_model, args.store_name)]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.mkdir(folder)


if __name__ == '__main__':
    main()
