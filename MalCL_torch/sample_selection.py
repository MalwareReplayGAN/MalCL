import torch
from torch.autograd import Variable
import torch.nn as nn


def common_vars(config, generator, classifier):

  # noise vector
  syn_size = math.ceil(((config.n_class-config.n_inc)*(config.k))/float(config.batchsize))
  #config.task
  #config.nb_batch
  synthetic = torch.tensor([]).to(config.device)
  logits_gen = torch.tensor([]).to(config.device)
  for i in range(syn_size):
    z_ = Variable(torch.rand((config.batchsize, config.z_dim)))
    if config.use_cuda:
      z_ = z_.to(config.device)

    generator.eval()
    with torch.no_grad():
      synthetic = torch.cat((synthetic, generator(z_)), dim=0)

  if config.sample_select == "L2_One_Hot":
    classifier.eval()
    with torch.no_grad():
      pred_label = classifier.predict(synthetic).detach() # Classifier predicts labels
    return synthetic, pred_label, None
  

  elif config.sample_select == "L1_B_Mean":
    classifier.eval()
    with torch.no_grad():
      pred_label = classifier.predict(synthetic).detach() # Classifier predicts labels
      for i in range(syn_size):
        synthetic_b = synthetic[i*config.batchsize:(i+1)*config.batchsize]
        logits_gen = torch.cat((logits_gen, classifier.get_logits(synthetic_b).to(config.device)), dim=0) # logit vector of synthetic samples
    if config.use_cuda:
      logits_gen = logits_gen.to(config.device)
    return synthetic, pred_label, logits_gen
  
  elif config.sample_select == "L1_C_Mean":
    classifier.eval()
    with torch.no_grad():
      for i in range(syn_size):
        synthetic_b = synthetic[i*config.batchsize:(i+1)*config.batchsize]
        logits_gen = torch.cat((logits_gen, classifier.get_logits(synthetic_b).to(config.device)), dim=0) # logit vector of synthetic samples
    if config.use_cuda:
      logits_gen = logits_gen.to(config.device)
    return synthetic, None, logits_gen
  return

################################ error massages ################################################

def err_msg_pred_label(config, cfg_sam_sel):
  if config.curr_classes-config.n_inc != len(cfg_sam_sel.pred_label[0]): 
    print("class number of prior task must be same with the number of pred_label columns")
    return None

def err_msg_logits_gen(config, cfg_sam_sel):
  if (config.curr_classes-config.n_inc)*config.k > len(cfg_sam_sel.logits_gen): 
    print("(class number of prior task)*k must be smaller than (or same with) the number of synthetic logits.")
    return None

def err_msg_logits_class(config, row_num):
  if config.curr_classes-config.n_inc != row_num: 
    print("class number of prior task must be same with the number of logits_class rows")
    return None
  
def err_msg_synthetic(config, cfg_sam_sel):
  if (config.curr_classes-config.n_inc)*config.k > len(cfg_sam_sel.synthetic): 
    print("(class number of prior task)*k must be smaller than (or same with) the number of synthetic samples")
    return None

def imbalance(config, for_one_hot):
  file_name = 'seed_'+str(config.seed_)+'_'+config.Generator_loss+'_epoch_'+str(config.epochs)+'_'+config.sample_select + "_imbalance"
  with open(file_name, 'a') as f:
    f.write(str(config.task)+'\n')
    for i in torch.unique(for_one_hot):
      f.write(str(i)+"\t")
    f.write("\n")

####################################### distances computations #######################################

def compute_L1(A, B):
  A = A.unsqueeze(1)
  B = B.unsqueeze(0)
  L1 = A - B
  L1 = torch.mean(torch.abs(L1), dim=2)
  return L1



def compute_L2(A, B):
  A = A.unsqueeze(1)
  B = B.unsqueeze(0)
  L2 = A - B
  L2 = torch.norm(L2, dim=2)
  return L2


####################################### selection methods ##########################################

def kNN_Sel(arr, k):
  k_NN_idx = []
  for i, row in enumerate(arr):   # iterations for rows (classes)
    for _ in range(k) : # select smallest k samples
      val, ind = torch.min(row, dim = 0) 
      k_NN_idx.append(ind.item()) 
      arr[:, ind] = torch.inf
  return k_NN_idx



def Glob_Sel(arr, sample_num):
  sorted_arr, sorted_index = torch.sort(torch.flatten(arr))
  cols_arr = sorted_index%len(arr[0])  # column positions
  uni, unique_idx = torch.unique(cols_arr, sorted=False, return_inverse=True)
  array_sort_id = uni.flip(0)[:sample_num]
  return array_sort_id


####################################### 3 types of sample selections ##########################################

# L2_One_Hot Scheme is k-NN selection based on L2 distances between the one-hot vector(ground truth) and softmax vector of synthetic samples
def L2_One_Hot(config, synthetic, pred_label, logits_gen=None, logits_real=None):
  one_hot = torch.eye(config.n_class-config.n_inc).to(config.device)
  arr = compute_L2(one_hot, pred_label)
  k_NN_idx = kNN_Sel(arr.to("cpu"), config.k)
  for_one_hot = torch.arange(0, config.n_class-config.n_inc).repeat_interleave(config.k)[:len(k_NN_idx)] # labels for synthetic samples
  label = nn.functional.one_hot(for_one_hot.to(torch.int64), num_classes = config.n_class) # turn into OneHot labels
  return torch.tensor(synthetic)[k_NN_idx].to(config.device), label.to(config.device)



# L1_B_Mean Scheme is global selection based on L1 distances between the logits of train set(mean batch) and logits of synthetic samples
def L1_B_Mean(config, synthetic, pred_label, logits_gen, logits_real):
  arr = compute_L1(logits_real.to(config.device), logits_gen)
  num_syn = len(arr[0])
  if num_syn < (config.n_class-config.n_inc)*config.k: sample_num = num_syn # set the number of selected samples
  else: sample_num = (config.n_class-config.n_inc)*config.k
  Glob_sort_id = Glob_Sel(arr.to("cpu"), sample_num) # perform Global selection based on L1 distances
  for_one_hot = torch.Tensor([list(i).index(max(i)) for i in torch.tensor(pred_label)[Glob_sort_id]]) # Classifier predicted label
  imbalance(config, for_one_hot)
  label = nn.functional.one_hot(for_one_hot.to(torch.int64), num_classes = config.n_class) # turn into OneHot labels
  return torch.tensor(synthetic)[Glob_sort_id].to(config.device), label.to(config.device)



# L1_C_Mean Scheme is k-NN selection based on L1 distances between the logits of train set(mean on classes) and logits of synthetic samples
def L1_C_Mean(config, synthetic, pred_label, logits_gen, logits_real):
  arr = compute_L1(logits_real.to(config.device), logits_gen)
  k_NN_idx = kNN_Sel(arr.to("cpu"), config.k)
  for_one_hot = torch.arange(0, config.n_class-config.n_inc).repeat_interleave(config.k)[:len(k_NN_idx)] # labels for synthetic samples
  label = nn.functional.one_hot(for_one_hot.to(torch.int64), num_classes = config.n_class) # turn into OneHot labels
  return torch.tensor(synthetic)[k_NN_idx].to(config.device), label.to(config.device)


