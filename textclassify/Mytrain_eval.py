# -*- coding:utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from datetime import timedelta

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


# 权重初始化，默认 xavier
def init_network(model, method = 'xavier', exclude='embedding', seed = 123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass




# 训练模型
def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    #模型设置成训练模式
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr = config.learning_rate)

    # 学习率指数衰减，每次epoch: 学习率 =  gamma * 学习率
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.9)

    total_batch = 0 #记录进行到多少epoch
    dev_best_loss = float('inf') #初始值为无限大
    last_improve = 0 # 记录上次验证集loss下降的batch数
    flag = False # 记录是否很久没有提升效果
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch+1, config.num_epochs))

        # scheduler.step() #学习率衰减
        for i , (trains, labels) in enumerate(train_iter):
            # 训练模型
            outputs = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                #每多少次输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu() #[batch_size, num_classes] 在列里面最大的数的索引，即分类所在的类别
                train_acc = metrics.accuracy_score(true, predic)

                # 验证集的准确率，损失值
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                #print('dev_acc')
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>5.2}, Train Acc: {2:>6.2%}, Val Loss: {3:>5.2}, Val Acc: {4:>6.2%}, Time: {5}{6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                model.train()

            total_batch += 1
            if total_batch - last_improve > config.requite_improvement:
                # 验证集loss 超过 1000 batch没下降，结束训练
                print('No optimization for a long time, auto-stopping ...')
                flag = True
                break
        if flag:
            break

    test(config, model, test_iter)


def test(config, model, test_iter):
    # 测试集
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss {0:>5.2}, Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print('Precision, Recall and F1-Score ...')
    print(test_report)
    print('Confusion Matrix ...')
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print('Time usage:', time_dif)


def evaluate(config, model, data_iter, test = False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([],dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            # 计算损失值
            loss = F.cross_entropy(outputs, labels)
            # 将损失值进行累加
            loss_total += loss
            # 转换成numpy
            labels = labels.data.cpu().numpy()
            # 计算出预测值
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            # 将真实的标签拼接一块
            labels_all = np.append(labels_all, labels)
            # 预测值拼接到一块
            predict_all = np.append(predict_all, predic)

    # 计算准确率
    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        '''
        classification_report(y_true, y_pred, target_names=target_names,digits=nums)
        y_true：1维数组，或标签指示器数组/稀疏矩阵，目标值。
        y_pred：1维数组，或标签指示器数组/稀疏矩阵，分类器返回的估计值。
        labels：array，shape = [n_labels]，报表中包含的标签索引的可选列表。
        target_names：字符串列表，与标签匹配的可选显示名称（相同顺序）。
        sample_weight：类似于shape = [n_samples]的数组，可选项，样本权重。
        digits：int，输出浮点值的位数．
        
        sklearn.metrics.confusion_matrix(y_true, y_pred, labels=None, sample_weight=None)
        y_true:实际的目标结果
        y_pred:预测的结果
        labels: 标签，对结果中的string进行排序， 顺序对应0、1、2
        sample_weight:样本的权重
        输出:
        一个矩阵，shape=[y中的类型数，y中的类型数]
        矩阵中每个值表征分类的准确性
        第0行第0列的数表示y_true中值为0，y_pred中值也为0的个数
        第0行第1列的数表示y_true中值为0，y_pred中值为1的个数
        
        '''
        return acc, loss_total / len(data_iter), report, confusion

    return acc, loss_total / len(data_iter)





