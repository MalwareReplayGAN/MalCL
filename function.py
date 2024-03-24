import torch
import numpy as np


def get_iter_dataset(x_train, y_train, task, nb_inc=None):
   if task is not None:
    if task == 0:
       selected_indices = np.where(y_train < init_classes)[0]
       return x_train[selected_indices], y_train[selected_indices]  
    else:
       start = init_classes + (task-1) * nb_inc
       end = init_classes + task * nb_inc
       selected_indices = np.where((y_train >= start) & (y_train < end))
       return x_train[selected_indices], y_train[selected_indices]
    

def run_batch(G, D, C, G_optimizer, D_optimizer, C_optimizer, x_, y_):
      x_ = x_.view([-1, feats_length])
      print("x_ shape", x_.shape) # [batchsize, feats_length] 16, 2381

      # y_real and y_fake are the label for fake and true data
      y_real_ = Variable(torch.ones(x_.size(0), 1))
      y_fake_ = Variable(torch.zeros(x_.size(0), 1))
      print("y_real_shape", y_real_.shape) # [batchsize, 1] 16, 1

      if use_cuda:
        y_real_, y_fake_ = y_real_.cuda(0), y_fake_.cuda(0)

      z_ = torch.rand((x_.size(0), z_dim))

      x_, z_ = Variable(x_), Variable(z_)

      if use_cuda:
        x_, z_, y_ = x_.cuda(0), z_.cuda(0), y_.cuda(0)

      # update D network
      D_optimizer.zero_grad()

      D_real = D(x_)
      print("D_real shape", D_real.shape) # [16, 1]
      print("y_real_[:x_.size(0)].shape: ", y_real_[:x_.size(0)].shape) # [16, 1]
      D_real_loss = BCELoss(D_real, y_real_[:x_.size(0)])

      G_ = G(z_)
      print('G_ shape', G_.shape) # 16, 2381
      D_fake = D(G_)
      print("D_fake shape", D_fake.shape) # 16, 1
      print("y_fake_[:x_.size(0)] shape", y_fake_[:x_.size(0)].shape) # 16, 1
      D_fake_loss = BCELoss(D_fake, y_fake_[:x_.size(0)])

      D_loss = D_real_loss + D_fake_loss

      D_loss.backward()
      D_optimizer.step()

      # update G network
      G_optimizer.zero_grad()

      G_ = G(z_)
      D_fake = D(G_)
      G_loss = BCELoss(D_fake, y_real_[:x_.size(0)])

      G_loss.backward()
      G_optimizer.step()

      # update C

      C_optimizer.zero_grad()
      print("y_ shape", y_.shape) # 16
      output = C(x_)
      if use_cuda:
         output = output.cuda(0)

      C_loss = criterion(output, y_)

      C_loss.backward()
      C_optimizer.step()

      return output

def get_replay_with_label(generator, classifier, batchsize, task, nb_inc):
    images_list = []
    labels_list = []
    task_label = [[] for _ in range(init_classes + (task-1) * nb_inc)]

    while True:
        if all(len(r) >= batchsize for r in task_label):
        # Checks whether there are at least 'batchsize' samples for each label in 'task_label'
        # The variable 'r' represents each innter list in 'task_label'
        # 'r' is a reference to one of the inner lists in 'task_label'
        # The loop continues until the condition is met for all inner lists, ensuring that each label has at least 'batchsize' samples.
            break
        z_ = Variable(torch.rand((batchsize, z_dim)))

        if use_cuda:
            z_ = z_.cuda(0)

        images = generator(z_)
        labels = classifier.predict(images)
        print(labels)
        
        for i in range(len(labels)):
            label = labels[i]
            # print(label)
            if (label < (init_classes + (task-1) * nb_inc)) and (len(task_label[label]) < batchsize):
                images_list.append(images[i].unsqueeze(0))
                labels_list.append(label.item())
                task_label[label].append(label.item())

        for i in range(len(task_label)):
            print("task_label:", i, "-", len(task_label[i]))
                

    images = torch.cat(images_list, dim=0)
    labels = torch.tensor(labels_list)

    return images.cpu(), labels.cpu()
