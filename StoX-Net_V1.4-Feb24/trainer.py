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
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import resnet
import dill

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# os.environ['NCCL_DEBUG'] = "INFO"
# os.environ['NCCL_IB_DISABLE'] = "1"

model_names = sorted(name for name in resnet.__dict__
                     if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(resnet.__dict__[name]))

parser = argparse.ArgumentParser(description='Property ResNets for CIFAR10 in pytorch')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20_1w1a',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet32)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,  # use different batch sizes
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',  # ./save_temp/StoX400BestEpoch_4w4a_AdaptTS1_NoSplit.th
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', default=False,
                    type=bool, help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='saved_models', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)
parser.add_argument('--max-time-steps', default=1, type=int, metavar='N',
                    help='Maximum time steps for each MTJ (default: 1)')
parser.add_argument('--max-ab', default=4, type=int, metavar='N',
                    help='Denotes maximum number of activation bits (default: 1)')
parser.add_argument('--max-wb', default=4, type=int, metavar='N',
                    help='Denotes maximum number of weight bits (default: 1)')

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

    model = torch.nn.DataParallel(resnet.resnet20_1w1a(abits=args.max_ab, wbits=args.max_wb))
    model.to('cuda')

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

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # train_loader = torch.utils.data.DataLoader(
    #     datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
    #         transforms.RandomHorizontalFlip(),
    #         transforms.RandomCrop(32, 4),
    #         transforms.ToTensor(),
    #         normalize,
    #     ]), download=False),
    #     batch_size=args.batch_size, shuffle=True,
    #     num_workers=args.workers, pin_memory=True, drop_last=True)
    #
    # val_loader = torch.utils.data.DataLoader(
    #     datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
    #         transforms.ToTensor(),
    #         normalize,
    #     ])),
    #     batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.workers, pin_memory=True, drop_last=True)

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./MNIST_Data', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomCrop(28, 4),
                                       transforms.ToTensor(),
                                       transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=args.batch_size, shuffle=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./MNIST_Data', train=False, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=args.batch_size, shuffle=False)

    print(f"Time to load: {time.time()-start_time}")
    # define loss function (criterion) and pptimizer
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
    #new logUp
    def Log_UP(t_min, t_max, epoch):
        return torch.tensor([t_min * math.pow(10, (math.log(t_max / t_min, 10) * (1.25 * ((epoch + 1) / args.epochs))))]).float().cuda()
    print(f"Time to setup helpers: {time.time()-start_time}")
    print(model.module)
    start_train = time.time()

    # ---------- Begin Importance Analysis ----------
    time_steps = [args.max_time_steps] * 18
    model.module.conv1.iterations = 1
    model.module.conv1.a_bits = 4
    model.module.conv1.w_bits = 4

    for i in range(3):
        model.module.layer1[i].conv1.iterations = max(time_steps[0 + (i * 6)], 1)
        model.module.layer1[i].conv2.iterations = max(time_steps[1 + (i * 6)], 1)
        model.module.layer2[i].conv1.iterations = max(time_steps[2 + (i * 6)], 1)
        model.module.layer2[i].conv2.iterations = max(time_steps[3 + (i * 6)], 1)
        model.module.layer3[i].conv1.iterations = max(time_steps[4 + (i * 6)], 1)
        model.module.layer3[i].conv2.iterations = max(time_steps[5 + (i * 6)], 1)

        model.module.layer1[i].conv1.a_bits = args.max_ab
        model.module.layer1[i].conv2.a_bits = args.max_ab
        model.module.layer2[i].conv1.a_bits = args.max_ab
        model.module.layer2[i].conv2.a_bits = args.max_ab
        model.module.layer3[i].conv1.a_bits = args.max_ab
        model.module.layer3[i].conv2.a_bits = args.max_ab

        model.module.layer1[i].conv1.w_bits = args.max_wb
        model.module.layer1[i].conv2.w_bits = args.max_wb
        model.module.layer2[i].conv1.w_bits = args.max_wb
        model.module.layer2[i].conv2.w_bits = args.max_wb
        model.module.layer3[i].conv1.w_bits = args.max_wb
        model.module.layer3[i].conv2.w_bits = args.max_wb
    # ---------- End Importance Analysis ----------

    # train for one epoch
    print("MTJ Samples: " + str(time_steps), '\n', "A Bits: " + str(args.max_ab), '\n', "W Bits: " + str(args.max_wb))

    print(f"Time to Start: {time.time()-start_time}")
    for epoch in range(args.start_epoch, args.epochs):
        start_epoch = time.time()
        t = Log_UP(T_min, T_max, epoch)
        if t < 1:
            k = 1 / t
        else:
            k = torch.tensor([1]).float().cuda()

        # model.module.conv1.k = k
        # model.module.conv1.t = t
        # layer_importance = []
        for i in range(3):

            model.module.layer1[i].conv1.k = k
            model.module.layer1[i].conv2.k = k
            model.module.layer1[i].conv1.t = t
            model.module.layer1[i].conv2.t = t
            # layer_importance.extend([model.module.layer1[i].importance1.item(), model.module.layer1[i].importance2.item()])

            model.module.layer2[i].conv1.k = k
            model.module.layer2[i].conv2.k = k
            model.module.layer2[i].conv1.t = t
            model.module.layer2[i].conv2.t = t
#             layer_importance.extend([model.module.layer2[i].importance1.item(), model.module.layer2[i].importance2.item()])

            model.module.layer3[i].conv1.k = k
            model.module.layer3[i].conv2.k = k
            model.module.layer3[i].conv1.t = t
            model.module.layer3[i].conv2.t = t
#             layer_importance.extend([model.module.layer3[i].importance1.item(), model.module.layer3[i].importance2.item()])

        # print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))

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
        start_t = time.time()
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        if args.half:
            input_var = input_var.half()
        variable_t = time.time() - start_t

        # compute output
        start_t = time.time()
        output = model(input_var)
        #output_t = time.time() - start_t

        start_t = time.time()
        loss = criterion(output, target_var)
        #loss_t = time.time() - start_t

        # compute gradient and do SGD step
        start_t = time.time()
        optimizer.zero_grad()
        loss.backward()
        # print("\nCOMPUTED BACKWARD\n")
        optimizer.step()
        backward_t = time.time() - start_t

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print('Variable Setup Time: {0}\t'
        #       'Output Time: {1}\t'
        #       'Loss Time: {2}\t'
        #       'Backward Time: {3}'.format(variable_t, output_t, loss_t, backward_t))

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
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
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

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

        # if i % args.print_freq == 0:
        #     print('Test: [{0}/{1}]\t'
        #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #           'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
        #               i, len(val_loader), batch_time=batch_time, loss=losses,
        #               top1=top1))

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
    # with torch.autograd.detect_anomaly():
    # torch.cuda.memory._record_memory_history()
    main()
