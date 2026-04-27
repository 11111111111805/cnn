import os
import sys
import argparse
import numpy as np
from datetime import datetime
import time
import matplotlib.pyplot as plt
import matplotlib

import pandas as pd
from sklearn.metrics import r2_score
from sklearn import metrics
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
sys.path.append('../')
from STDenseNetFusion.dataloader.milano_crop import load_data
from STDenseNetFusion.models.DenseNet_v2 import DenseNet


#检查PyTorch安装是否支持CUDA
#import torch
#print(torch.cuda.is_available())


torch.manual_seed(22)

parse = argparse.ArgumentParser()
parse.add_argument('-height', type=int, default=20)                    #图像高度，默认20
parse.add_argument('-width', type=int, default=20)                     #图像宽度，默认20
parse.add_argument('-traffic', type=str, default='sms')                #流量数据类型默认sms
parse.add_argument('-close_size', type=int, default=3)                  #接近性数据大小默认为3，模型的接近性数据的特征数量
parse.add_argument('-period_size', type=int, default=3)                 #周期性数据大小默认为3，模型的周期性数据的特征数量
parse.add_argument('-trend_size', type=int, default=0)                  #趋势性数据大小默认为0，模型的趋势性数据的特征数量
parse.add_argument('-test_size', type=int, default=24*7)                #测试集大小，默认为一周的小时数，24*7
parse.add_argument('-nb_flow', type=int, default=2)                     #流的数量，默认为2，模型中流的数量
parse.add_argument('-crop', dest='crop', action='store_true')
parse.add_argument('-no-crop', dest='crop', action='store_false')
parse.set_defaults(crop=False)                                          #是否裁剪图像的标志，默认f
parse.add_argument('-train', dest='train', action='store_true')
parse.add_argument('-no-train', dest='train', action='store_false')
parse.set_defaults(train=True)                                          #是否进行训练的标志，默认t
parse.add_argument('-rows', nargs='+', type=int, default=[5, 15])      #行数的列表，默认为【5,15】，在图像中选择的行的范围
parse.add_argument('-cols', nargs='+', type=int, default=[5, 15])      #列数的列表，默认为【5,15】，在图像中选择的列的范围
parse.add_argument('-loss', type=str, default='l2', help='l1 | l2')     #损失函数类型，默认为l2（分别有l1损失和l2损失）
parse.add_argument('-lr', type=float, default=0.002)               #学习率，默认为0.001
parse.add_argument('-batch_size', type=int, default=32, help='batch size')  #批量大小，默认为32，模型训练时每个批次的样本数量
parse.add_argument('-epoch_size', type=int, default=100, help='epochs')     #迭代次数，默认为100.模型训练时的迭代轮数
parse.add_argument('-drop_rate', type=float, default=0.2, help='dropoutrate')
parse.add_argument('-test_row', type=int, default=10, help='testrow')      #测试行数，默认为51，测试中图像的行号
parse.add_argument('-test_col', type=int, default=19, help='testcol')      #测试列数，默认为60，测试的图像中的列号
parse.add_argument('-save_dir', type=str, default='../results')            #保存结果的目录，默认为result。保存模型训练和测试结果的目录路径

opt = parse.parse_args()                    #解析命令行参数，并将解析结果存储在opt中
# print(opt)                                #打印各项参数
# opt.save_dir = '{}/{}'.format(opt.save_dir, opt.traffic)
opt.model_filename = '{}/model={}-loss={}-lr={}-close={}-period=' \
                     '{}-trend={}'.format(opt.save_dir,
                                          'densenet',
                                          opt.loss, opt.lr, opt.close_size,
                                          opt.period_size, opt.trend_size)
#将新的 opt.save_dir、模型类型、损失函数类型、学习率、接近性数据大小、周期性数据大小、趋势性数据大小信息拼接在一起，形成模型文件的名称
# print('Saving to ' + opt.model_filename)          #打印文件名


#查看result目录
#import os
#current_dir = os.getcwd()                               # 获取当前工作目录
#results_dir = os.path.join(current_dir, "results")      # 拼接得到 "results" 目录的完整路径
#print("Results directory:", results_dir)

# 设置Matplotlib的默认字体为微软雅黑
matplotlib.rcParams['font.family'] = 'Microsoft YaHei'
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号无法显示的问题


def log(fname, s):              #日志函数log，向指定文件写入日志信息，写入文件名fname和字符串s
    if not os.path.isdir(os.path.dirname(fname)):           #os.path.dirname(fname)获取fname父目录路径，os.path.isdir判断该路径是否是一个目录。如果该路径不是一个目录，则进入条件语句
        os.system("mkdir -p " + os.path.dirname(fname))     #使用系统命令mkdir -p创建目录，-p参数表示如果目录不存在则创建，如果目录已经存在则不做操作。它的作用是创建 fname 的父目录，以确保写入日志的目录存在
    f = open(fname, 'a')                                    #打开文件fname，如若文件不存在则自动创建
    f.write(str(datetime.now()) + ': ' + s + '\n')          #将当前datetime.now()转换为字符串，
    f.close()


def set_lr(optimizer, epoch, n_epochs, lr):     #设置学习率函数，根据训练的当前轮次和总轮次调整学习率，四个参数：优化器opt，当前轮次epo，总轮次n_epo，初始学习率lr
    #lr = lr    这行好像没啥意义
    if float(epoch) / n_epochs > 0.75:          #检查当前轮次占总轮次的比例是否超过了 0.75
        lr = lr * 0.01                          #学习率 lr 乘以 0.01，相当于将学习率缩小为原来的 1%，以降低学习速率
    if float(epoch) / n_epochs > 0.5:           #检查当前轮次占总轮次的比例是否超过了 0.5
        lr = lr * 0.1                           #学习率 lr 乘以 0.1，相当于将学习率缩小为原来的 10%，以降低学习速率

    for param_group in optimizer.param_groups:  #遍历优化器optimizer的参数组
        param_group['lr'] = lr                  #将学习率 lr 赋值给优化器中的每个参数组的学习率

    return lr


def train_epoch(data_type='train'):         #用于训练一个轮次的数据，datatype决定针对训练集还是验证集进行训练
    total_loss = 0                          #初始化总损失为0
    if data_type == 'train':                #对训练集训练
        model.train()                       #训练模式（python内置函数）
        data = train_loader                 #将数据加载器设置为训练数据加载器
    if data_type == 'valid':                #对验证集进行训练
        model.eval()                        #评估模式（不进行梯度计算）
        data = valid_loader                 #将数据加载器设置为验证数据加载器

    # 根据数据类型和特征大小执行不同的训练过程
    if (opt.period_size > 0) & (opt.close_size > 0) & (opt.trend_size > 0):     #同时存在周期，接近，趋势性数据
        for idx, (c, p, t, target) in enumerate(data):
            optimizer.zero_grad()                                               #将优化器中的梯度清零，便于接受新的梯度
            model.zero_grad()                                                   #将模型中的梯度清零，确保不会再当前轮次内的反向传播过程中积累之前批次的梯度信息
            # 笔记本记录
            input_var = [Variable(_.float()).cuda() for _ in [c, p, t]]
            target_var = Variable(target.float(), requires_grad=False).cuda()

            pred = model(input_var)                 #调用模型进行预测，对输入数据进行预测，得到预测结果pred
            loss = criterion(pred, target_var)      #损失函数criterion计算模型预测值与目标值之间损失
            total_loss += loss.item()               #累加总损失
            if data_type == 'train':
                loss.backward()                     #反向传播，计算损失对于模型参数的梯度
                optimizer.step()                    #使用优化器 optimizer 根据梯度信息更新模型的参数，以最小化损失
    elif (opt.close_size > 0) & (opt.period_size > 0):                          #存在接近，周期性数据
        for idx, (c, p, target) in enumerate(data):
            optimizer.zero_grad()
            model.zero_grad()
            input_var = [Variable(_.float()).cuda() for _ in [c, p]]
            target_var = Variable(target.float(), requires_grad=False).cuda()

            pred = model(input_var)
            loss = criterion(pred, target_var)
            total_loss += loss.item()
            if data_type == 'train':
                loss.backward()
                optimizer.step()
    elif opt.close_size > 0:                                                    #存在接近性数据
        for idx, (c, target) in enumerate(data):
            optimizer.zero_grad()
            model.zero_grad()
            x = [Variable(c.float()).cuda()]
            y = Variable(target.float(), requires_grad=False).cuda()

            pred = model(x)
            loss = criterion(pred, y)
            total_loss += loss.item()
            if data_type == 'train':
                loss.backward()
                optimizer.step()

    return total_loss


def train():            #训练模型
    os.system("mkdir -p " + opt.save_dir)                       #创建保存模型和日志的目录
    best_valid_loss = 1.0
    train_loss, valid_loss = [], []                             #用于存储每个epoch的训练损失和验证损失
    for i in range(opt.epoch_size):
        train_loss.append(train_epoch('train'))                 #调用train_epoch('train')对训练集训练，并将损失添加
        valid_loss.append(train_epoch('valid'))                 #调用train_epoch('valid')对训练集训练，并将损失添加
        scheduler.step()

        if valid_loss[-1] < best_valid_loss:  # 如果当前验证损失小于最佳验证损失，则更新最佳验证损失，并保存模型和优化器的状态
            best_valid_loss = valid_loss[-1]

            torch.save({'epoch': i, 'model': model, 'train_loss': train_loss,
                        'valid_loss': valid_loss}, opt.model_filename + '.model')  # 将epoch数，模型，训练损失，验证损失等信息字典保存到文件中
            torch.save(optimizer, opt.model_filename + '.optim')  # 保存优化器状态

        log_string = ('iter: [{:d}/{:d}], train_loss: {:0.6f}, valid_loss: {:0.6f}, '
                      'best_valid_loss: {:0.6f}, lr: {:0.5f}').format((i + 1), opt.epoch_size,
                                                                      train_loss[-1],
                                                                      valid_loss[-1],
                                                                      best_valid_loss,
                                                                      opt.lr)  # 记录当前的迭代信息，epoch 数、训练损失、验证损失、历史最佳验证损失和当前学习率
        print(log_string)
        log(opt.model_filename + '.log', log_string)  # 写入日志文件中，记录训练信息

    #loss_data = pd.DataFrame(train_loss)  # 将train_loss列表转换为pandas的DataFrame对象，便于数据处理和保存
    #loss_data.to_csv(
    #    r'D:\project\python\traffic_prediction-master\csv\val_result_{}_{}_loss_100.csv'.format(str(opt.test_row),
    #                                                                                            str(opt.test_col)),
    #    index=False)
    # 将DataFrame对象保存为csv文件

from sklearn.metrics import r2_score

def predict(test_type='train'):
    predictions = []        # 初始化列表，存储预测结果
    ground_truth = []       # 初始化列表，存储真实值
    loss = []               # 初始化列表，存储损失
    best_model = torch.load(opt.model_filename + '.model').get('model')  # 加载最佳模型的状态字典，并提取模型对象model

    if test_type == 'train':            #使用训练集预测
        data = train_loader
    elif test_type == 'test':           #使用测试集预测
        data = test_loader
    elif test_type == 'valid':          #使用验证集预测
        data = valid_loader

    if (opt.period_size > 0) & (opt.close_size > 0) & (opt.trend_size > 0):         #数据包括周期，接近，趋势性数据
        for idx, (c, p, t, target) in enumerate(data):
            input_var = [Variable(_.float()).cuda() for _ in [c, p, t]]
            target_var = Variable(target.float(), requires_grad=False).cuda()
            pred = best_model(input_var)                        #使用最佳模型对输入变量进行预测，预测结果pred
            predictions.append(pred.data.cpu().numpy())         #把预测结果转换为CPU的numpy数组，并添加到predictions列表中
            ground_truth.append(target.numpy())                 #目标函数转换为numpy数组，添加到ground_truth列表中
            loss.append(criterion(pred, target_var).item())     #计算预测结果与目标数据之间的损失，损失值添加到loss中
    elif (opt.close_size > 0) & (opt.period_size > 0):
        for idx, (c, p, target) in enumerate(data):
            input_var = [Variable(_.float()).cuda() for _ in [c, p]]
            target_var = Variable(target.float(), requires_grad=False).cuda()
            pred = best_model(input_var)
            predictions.append(pred.data.cpu().numpy())
            ground_truth.append(target.numpy())
            loss.append(criterion(pred, target_var).item())
    elif opt.close_size > 0:
        for idx, (c, target) in enumerate(data):
            input_var = Variable(c.float()).cuda()
            target_var = Variable(target.float(), requires_grad=False).cuda()
            pred = best_model(input_var)
            predictions.append(pred.data.cpu().numpy())
            ground_truth.append(target.numpy())
            loss.append(criterion(pred, target_var).item())

    final_predict = np.concatenate(predictions)     #连接所有预测结果，得到包含所有预测结果的数组final
    ground_truth = np.concatenate(ground_truth)     #连接所有真实值，得到包含所有真实值的数组ground

    rmse = []
    for y_hat, y in zip(final_predict, ground_truth):
        flows, height, width = y_hat.shape
        y_hat = np.reshape(y_hat, (flows, height * width)) * (mmn.max - mmn.min)
        y = np.reshape(y, (flows, height * width)) * (mmn.max - mmn.min)
        rmse.append(metrics.mean_squared_error(y_hat, y) ** 0.5)
    print(test_type + ' RMSE:{:0.5f}'.format(np.mean(rmse)))

    #r2_scores = []
    #for y_hat1, y1 in zip(final_predict, ground_truth):
    #    r2 = r2_score(y1, y_hat1)
    #    r2_scores.append(r2)
    #print(test_type + ' R-squared (R2) Score:{:0.5f}'.format(np.mean(r2_scores)))

    if opt.test_row & opt.test_col:             #如果opt.test_row和opt.test_col均为真（非零），则使用它们作为行和列的索引
        row, col = opt.test_row, opt.test_col
    else:                                       #否则使用ground_truth数组的形状信息计算中间行和列的索引
        row_length, col_length = ground_truth.shape[-2:]
        row, col = int(row_length / 2), int(col_length / 2)

    plt.figure()                                #在matplotlib创建一个新的图形对象
    # 绘制预测值和真实值的曲线图，并乘以缩放因子还原为初始值
    plt.plot(final_predict[:, 0, row, col] * (mmn.max - mmn.min), 'r-', label='Predicted')
    plt.plot(ground_truth[:, 0, row, col] * (mmn.max - mmn.min), 'k-', label='GroundTruth')
    plt.legend(loc='upper right')
    plt.savefig('../results/predictions.png')   #保存为predictions
    # plt.show()


#将数据集分割训练集和验证集，输入参数数据加载器 dataloader、测试集大小 test_size、是否打乱数据顺序 shuffle，以及随机种子 random_seed
def train_valid_split(dataloader, test_size=0.2, shuffle=True, random_seed=0):
    length = len(dataloader)
    indices = list(range(0, length))

    if shuffle:                         #shuffle为1则打乱索引列表
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    if type(test_size) is float:        #如果 test_size 是浮点数，则将其乘以数据数量并向下取整，得到测试集的大小
        split = int(np.floor(test_size * length))
    elif type(test_size) is int:        #如果 test_size 是整数，则直接作为测试集的大小
        split = test_size
    else:                               #如果 test_size 既不是浮点数也不是整数，则抛出 ValueError
        raise ValueError('%s should be an int or float'.format(str))
    return indices[split:], indices[:split]     #根据分割后的索引位置，返回训练集和验证集的索引列表


if __name__ == '__main__':
    #加载数据集，并进行预处理
    path = './data/all_data_sliced.h5'
    x_train, y_train, x_test, y_test, mmn = load_data(path, opt.traffic, opt.close_size, opt.period_size,
                                                      opt.trend_size,
                                                      opt.test_size, opt.nb_flow, opt.height, opt.width, opt.crop,
                                                      opt.rows, opt.cols)
    x_train.append(y_train)
    x_test.append(y_test)
    train_data = list(zip(*x_train))
    test_data = list(zip(*x_test))
    print(len(train_data), len(test_data))
    start_time = time.time()

    #分割训练集和数据集
    train_idx, valid_idx = train_valid_split(train_data, 0.1, shuffle=True)
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    # 创建数据加载器，用于模型训练和评估过程中加载数据
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, sampler=train_sampler, pin_memory=True)
    valid_loader = DataLoader(train_data, batch_size=opt.batch_size, sampler=valid_sampler, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False)

    #channels列表用于指定每个数据通道的特征数量，由流和不同类型数据大小决定
    channels = [opt.close_size * opt.nb_flow,       #接近性数据的特征数量，3 * 2 = 6
                opt.period_size * opt.nb_flow,      #周期性数据的特征数量，3 * 2 = 6
                opt.trend_size * opt.nb_flow]       #趋势性数据的特征数量，0 * 2 = 0
    model = DenseNet(nb_flows=opt.nb_flow, drop_rate=opt.drop_rate, channels=channels).cuda()   #创建模型
    optimizer = optim.Adam(model.parameters(), opt.lr)      #优化器优化模型参数

    #optim.lr_scheduler.MultiStepLR方法来设置学习率的调度策略，根据迭代轮次变化学习率
    #使用阶梯式衰减
    #当迭代轮次达到 0.5 * opt.epoch_size 和 0.75 * opt.epoch_size 时，学习率会分别乘以 0.1
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.5 * opt.epoch_size),
                                                                      int(0.75 * opt.epoch_size)], gamma=0.1)
    # optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9)
    # print(model)

    # 检查保存模型和日志的目录是否存在，如果不存在则创建该目录
    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)
    if not os.path.isdir(opt.save_dir):
        raise Exception('%s is not a dir' % opt.save_dir)

    # 选择相应的损失函数，选择 L1损失函数nn.L1Loss()，如果是l2，选择均方误差损失函数nn.MSELoss()
    if opt.loss == 'l1':
        criterion = nn.L1Loss().cuda()
    elif opt.loss == 'l2':
        criterion = nn.MSELoss().cuda()

    print('Training...')
    # 记录训练过程的日志
    log(opt.model_filename + '.log', '[training]')
    # 如果opt.train为1，则进行模型训练
    if opt.train:
        train()

    #加载模型优化器“optimizer”
    model = torch.load(opt.model_filename + '.optim')
    predict('test')
    end_time = time.time()
    run_time = end_time - start_time
    print("模型运行时间：{}秒".format(run_time))
    #绘图
    train_loss = torch.load(opt.model_filename + '.model').get('train_loss')[1:-1]
    test_loss = torch.load(opt.model_filename + '.model').get('valid_loss')[:-1]

    #plt.figure()
    #plt.plot(train_loss, 'r-',label = "训练损失")
    #plt.plot(test_loss,'k-',label = "测试损失")
    #plt.legend(loc = 'best')
    #plt.savefig('../results/combine_train_test_loss.png')

    plt.figure()
    plt.plot(torch.load(opt.model_filename + '.model').get('train_loss')[1:-1], 'r-')
    plt.legend(labels=['train_loss'], loc='best')
    plt.savefig('../results/train_loss.png')

    plt.figure()
    plt.plot(torch.load(opt.model_filename + '.model').get('valid_loss')[:-1], 'k-')
    plt.legend(labels=['test_loss'], loc='best')
    plt.savefig('../results/test_loss.png')
    #plt.show()
