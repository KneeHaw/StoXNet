import argparse
import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import resnet
import utilities
import dill
import argsparser

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
parser = argsparser.get_parser()
best_prec1 = 0
Log_Vals = open("./saved_logs/test.txt", 'w')
save_name = 'test.th'


def main():
    print("Entering Model")
    start_time = time.time()

    global args, best_prec1
    args = parser.parse_args()

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    print(f"Time for args: {time.time() - start_time}")

    model = resnet.resnet20_1w1a(args.num_ab, args.num_ab, args.ab_sw, args.wb_sw, args.subarray_size, args.time_steps)
    model.to("cuda")
    print(f"Model Loaded: {time.time()-start_time}")

    # optionally resume from a checkpoint
    if args.resume and args.evaluate:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = False

    print(f"Loading Data: {time.time()-start_time}")
    train_loader, val_loader = utilities.get_loaders(dataset=args.dataset, batch_size=args.batch_size, workers=4)
    print(f"Time to load: {time.time()-start_time}")

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    if args.half:
        model.half()
        criterion.half()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                               momentum=args.momentum,
                              weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0, last_epoch=-1)

    if args.arch in ['resnet1202', 'resnet110']:
        # for resnet1202 original paper uses lr=0.01 for first 400 minibatches for warm-up
        # then switch back. In this implementation it will correspond for first epoch.
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr*0.1

    T_min, T_max = 1e-3, 1e1

    def Log_UP(t_min, t_max, epoch):
        return torch.tensor([t_min * math.pow(10, (math.log(t_max / t_min, 10) * (1.25 * ((epoch + 1) / args.epochs))))]).float().cuda()

    print(f"Time to setup helpers: {time.time()-start_time}")
    print(model.modules)  # Print all model components/layers
    start_train = time.time()

    # train for one epoch
    print(f"Time to Start: {time.time()-start_time}")

    for epoch in range(args.start_epoch, args.epochs):
        start_epoch = time.time()
        t = Log_UP(T_min, T_max, epoch)
        if t < 1:
            k = 1 / t
        else:
            k = torch.tensor([1]).float().cuda()

        for i in range(3):
            model.layer1[i].conv1.k = k
            model.layer1[i].conv2.k = k
            model.layer1[i].conv1.t = t
            model.layer1[i].conv2.t = t

            model.layer2[i].conv1.k = k
            model.layer2[i].conv2.k = k
            model.layer2[i].conv1.t = t
            model.layer2[i].conv2.t = t

            model.layer3[i].conv1.k = k
            model.layer3[i].conv2.k = k
            model.layer3[i].conv1.t = t
            model.layer3[i].conv2.t = t

        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))

        if args.evaluate is True:
            validate(val_loader, model, criterion)
            exit()

        train(train_loader, model, criterion, optimizer, epoch)
        lr_scheduler.step()

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if is_best:
            save_checkpoint({
                'model': model,
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(args.save_dir, save_name))
        print("Epoch Time: " + str(time.time() - start_epoch))
        Log_Vals.write(str(prec1) + '\n')
    print("Total Time: " + str(time.time() - start_train))

    # save_checkpoint({
    #     'state_dict': model.state_dict(),
    #     'best_prec1': best_prec1,
    # }, is_best, filename=os.path.join(args.save_dir, 'StoX50Epoch.th'))


def train(train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = torch.autograd.Variable(input.cuda())
        target_var = torch.autograd.Variable(target.cuda())
        if args.half:
            input_var = input_var.half()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        # prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        # top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch+1, i+1, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))
    Log_Vals.write(str(epoch+1) + ', ' + str(losses.avg) + ', ' + str(top1.avg) + ', ')


def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        with torch.no_grad():
            input_var = torch.autograd.Variable(input.cuda())
            target_var = torch.autograd.Variable(target.cuda())

        # if args.half:
        #     input_var = input_var.half()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Save the training model"""
    torch.save(state, filename, pickle_module=dill)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
