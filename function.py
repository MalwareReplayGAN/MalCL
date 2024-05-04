import torch
from torch.autograd import Variable
import numpy as np
import pandas
from sklearn.preprocessing import StandardScaler
from data_ import oh
import time


def get_iter_dataset(x_train, y_train, batchsize, task, init_classes=None, nb_inc=None):
   
   if task is not None:
    if task == 0:
       selected_indices = np.where(y_train < init_classes)[0] 
    else:
       start = init_classes + (task-1) * nb_inc
       end = init_classes + task * nb_inc
       selected_indices = np.where((y_train >= start) & (y_train < end))
    
    x_, y_ = x_train[selected_indices], y_train[selected_indices]

    # Manage Class Imbalance Issue
    class_sample_count = np.array([len(np.where(y_ == t)[0]) for t in np.unique(y_)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in y])
    samples_weight = torch.from_numpy(samples_weight).float()
    sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
    
    x_ = torch.from_numpy(x_).type(torch.FloatTensor)
    y_ = torch.from_numpy(y_).type(torch.FloatTensor)

    # Scaling
    scaler = StandardScaler()
    x_ = scaler.fit_transform(x_)
    x_ = torch.FloatTensor(x_)
    
    # One-hot Encoding
    y_oh = oh(y_, num_classes=init_classes+nb_inc*task)
    y_oh = torch.Tensor(y_oh)

    data_tensored = torch.utils.data.TensorDataset(x_, y_oh)
    trainLoader = torch.utils.data.DataLoader(data_tensored, batch_size=batchsize, num_workers=1, sampler=sampler, drop_last=True)

    return trainLoader


def ground(a):
    new = np.zeros((a, a))
    for i in range(a):
        new[i][i] = 1
    return torch.Tensor(new)


def Rank(sumArr, img, y1, k):
    start = time.time()
    img_list = img.tolist()
    y1_list = y1.tolist()
    zip(img_list, y1_list)
#    print("time for to list", time_for_to_list = time.time()-start)
    y = pandas.DataFrame({'a': sumArr, 'b':img_list, 'c':y1_list})
#    print("time for dataframe", time_for_dataframe = time.time()-start - time_for_to_list)
    y = y.sort_values(by=['a'], axis = 0)
    img_ = y['b'][0:k]
    y1_ = y['c'][0:k]
    return img_.tolist(), y1_.tolist()


def GetL2Dist(y1, y2):
    sumArr = []
    for i in range(len(y1)):
        arr = []
        for (a, b) in zip(y1[i].tolist() ,y2.tolist()):
            arr.append((a-b)**2)
        sumArr.append(sum(arr))
    return sumArr


def selector(images, label, k):
    img = []
    lbl = []
    lbl_for_one_hot = []
    GroundTruth = ground(len(label[0]))
    for i in range(len(GroundTruth)):
        sumArr = GetL2Dist(label, GroundTruth[i])
        new_images, new_label = Rank(sumArr, images, label, k)
        img = img + new_images
        lbl = lbl + new_label
    for k in lbl:
      lbl_for_one_hot.append(k.index(max(k)))
    return torch.tensor(img), torch.tensor(lbl_for_one_hot)
