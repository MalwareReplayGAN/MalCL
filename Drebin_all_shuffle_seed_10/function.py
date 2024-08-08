import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import pandas
from sklearn.preprocessing import StandardScaler
from drebin_data_ import oh
import time



def get_iter_train_dataset(x, y, n_class=None, n_inc=None, task=None):
   
   if task is not None:
    if task == 0:
       selected_indices = np.where(y < n_class)[0] 
    else:
       start = n_class - n_inc
       end = n_class
       selected_indices = np.where((y >= start) & (y < end))
    
    return x[selected_indices], y[selected_indices]



def get_iter_test_dataset(x, y, n_class):
    selected_indices = np.where(y < n_class)[0] 
    return x[selected_indices], y[selected_indices]



def get_dataloader(x, y, batchsize, n_class, scaler):

    # Manage Class Imbalance Issue
    y_ = np.array(y, dtype=int)
    class_sample_count = np.array([len(np.where(y_ == t)[0]) for t in np.unique(y_)])
    weight = 1. / class_sample_count
    weight = 1. / class_sample_count
    min_ = (min(np.unique(y_)))
    samples_weight = np.array([weight[t-min_] for t in y_])
    
    samples_weight = torch.from_numpy(samples_weight).float()
    sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
    
    x_ = torch.from_numpy(x).type(torch.FloatTensor)
    y_ = torch.from_numpy(y_).type(torch.FloatTensor)

    # Scaling
    scaler = scaler.partial_fit(x_)
    x_ = scaler.transform(x_)
    x_ = torch.FloatTensor(x_)
    
    # One-hot Encoding
    y_oh = oh(y_, num_classes=n_class)
    y_oh = torch.Tensor(y_oh)

    data_tensored = torch.utils.data.TensorDataset(x_, y_oh)
    trainLoader = torch.utils.data.DataLoader(data_tensored, batch_size=batchsize, num_workers=1, sampler=sampler, drop_last=True)

    return trainLoader, scaler



def ground(a):
    new = np.zeros((a, a))
    for i in range(a):
        new[i][i] = 1
    return torch.Tensor(new)

def duplicate_index(index, c):
    for i in index:
        if c in i:
            return True
    return False

def Rank(sumArr, img, y1, k, index_):

    img_ = []
    y1_ = []
    id = []

#    start = time.time()
    index = [i for i in range(len(y1))]
    img_list = img.tolist()
    y1_list = y1.tolist()
    y = pandas.DataFrame({'a': sumArr, 'b':img_list, 'c':y1_list, 'd':index})
    y = y.sort_values(by=['a'], axis = 0)

    for i in range(len(y['b'])):
        if len(id) == k:
            break
        if duplicate_index(index_, y['d'][i]):
            continue
        
        img_.append(y['b'][i])
        y1_.append(y['c'][i])
        id.append(y['d'][i])

    return img_, y1_, id

def duplicate(index):
    count = 0
    #print("index", index)
    for i in range(len(index)):
        for j in range(i+1, len(index)):
              count+=len(set(index[i]).intersection(set(index[j])))
    return count

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
    index = []
    GroundTruth = ground(len(label[0]))

    new_f = open('duplicate', 'a')
    for i in range(len(GroundTruth)):
        sumArr = GetL2Dist(label, GroundTruth[i])
        new_images, new_label, new_index = Rank(sumArr, images, label, k, index)
        img = img + new_images
        lbl = lbl + new_label
        index = index + [new_index]
    duplicate_count = duplicate(index)
    new_f.write(str(duplicate_count))
    new_f.write('\t')
    new_f.close()
    for k in lbl:
      lbl_for_one_hot.append(k.index(max(k)))
    return torch.tensor(img), torch.tensor(lbl_for_one_hot)



def test(model, x_test, y_test, n_class, device, scaler):

    x_test = scaler.transform(x_test)
    x_test = torch.FloatTensor(x_test)
    y_test = torch.Tensor(y_test)
    y_test = y_test.float()

    Y_test_onehot = oh(y_test, num_classes=n_class)
    Y_test_onehot = torch.Tensor(Y_test_onehot)
    Y_test_onehot = Y_test_onehot.float()

    x_test = x_test.to(device)
    Y_test_onehot = Y_test_onehot.to(device)
    y_test = y_test.to(device)

    model.eval()
    prediction = model(x_test)
    predicted_classes = prediction.max(1)[1]
    correct_count = (predicted_classes == y_test).sum().item()
    cost = F.cross_entropy(prediction, Y_test_onehot)
    accuracy = correct_count / len(y_test) * 100
    
    print('Accuracy: {}% Cost: {:.6f}'.format(
        accuracy, cost.item()
    ))

    return accuracy
       
