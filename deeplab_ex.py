import os
import torch
import torchvision
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import cv2
import torchvision.models as models
from tensorboardX import SummaryWriter
from datasets_ex import get_loader
from utils.utils import combine2tensor
import time

class Loss(nn.Module):
    def __init__(self, weight=[1.0] * 7):
        super(Loss, self).__init__()
        self.weight = weight

    '''
    def forward(self, pred, label):
        n, c, h, w = pred.shape
        nt, ct, ht, wt = label.shape 
        #pred = pred.view(n*w*h,-1)
        #label = label.view(n*w*h, -1)
        loss = F.binary_cross_entropy_with_logits(pred, label)
        return loss/n
    '''
    
    def forward(self, input, target):
        n, c, h, w = input.size()
        # assert(max(target) == 1)
        log_p = input.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        target_t = target.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        target_trans = target_t.clone()
        pos_index = (target_t >0)
        neg_index = (target_t ==0)
        target_trans[pos_index] = 1
        target_trans[neg_index] = 0
        pos_index = pos_index.data.cpu().numpy().astype(bool)
        neg_index = neg_index.data.cpu().numpy().astype(bool)
        weight = torch.Tensor(log_p.size()).fill_(0)
        weight = weight.numpy()
        pos_num = pos_index.sum()
        neg_num = neg_index.sum()
        sum_num = pos_num + neg_num
        weight[pos_index] = neg_num*1.0 / sum_num
        weight[neg_index] = pos_num*1.0 / sum_num

        weight = torch.from_numpy(weight)
        weight = weight.cuda()
        loss =F.binary_cross_entropy_with_logits(log_p, target_t , weight, size_average=True)
        return loss

class Trainer(object):
    def __init__(self):

        torch.backends.cudnn.benchmark = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.loss = Loss()
        self.criterion = Loss().to(self.device)

        self.model = models.segmentation.deeplabv3_resnet50(pretrained=False,num_classes = 1) #the shoulder og Giant otherwise 
                                                                       #you need contruct the model in model.deeplabv3
        self.model = self.model.to(self.device)
        
        #if self.gpu_ids is not None:
        #    self.model = torch.nn.DataParallel(self.model, device_ids=self.gpu_ids)
        
        #for pretrain
        self.resume = False
        self.checkpoint_dir = None
        if self.resume: #or not self.opt.isTrain:
            trained_dict = torch.load(self.checkpoint_dir)
            model_dict = self.model.state_dict()
            trained_dict = {k: v for k, v in trained_dict.items() if k in model_dict}
            model_dict.update(trained_dict)
            self.model.load_state_dict(model_dict)
            print('network resumed')
        else:
            print('network built!')
        

       
        self.writer = SummaryWriter()
        self.lr = 0.0003
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                          lr=self.lr)


        ##parament for datset MSRA
        self.num_epochs = 500
        self.img_root = '/home/liuyang/Documents/data/MSRA/images'
        self.label_root = '/home/liuyang/Documents/data/MSRA/labels'
        self.batch_size = 8
        self.img_size = 224 
        self.dataset = get_loader(self.img_root, self.label_root, img_size = self.img_size, batch_size = self.batch_size)
        self.dataset_size = self.dataset.__len__()
    @staticmethod
    def adjust_learning_rate(optimizer, decay_rate=0.9):

        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay_rate
    
    
    def train(self):
        self.model.train() # (set in training mode, this affects BatchNorm and dropout)
        for epoch in range(self.num_epochs):
            train_loss = 0
            if epoch in [100, 300, 500]:
                self.adjust_learning_rate(optimizer=self.optimizer, decay_rate=0.1)
            print ("###########################")
            print ("######## NEW EPOCH ########")
            print ("###########################")
            print ("epoch: %d/%d" % (epoch+1, self.num_epochs))

            for batch_id, (imgs, label_imgs) in enumerate(self.dataset):
                global_step = batch_id + epoch * self.dataset_size
                current_time = time.time()

                imgs =  torch.autograd.Variable(imgs).to(self.device) # (shape: (batch_size, 3, img_h, img_w))
                label_imgs =  torch.autograd.Variable(label_imgs).to(self.device) # (shape: (batch_size, 1, img_h, img_w))

                outputs = self.model(imgs) # (shape: (batch_size, cls_num, img_h, img_w))

            # compute the loss:
                loss = self.criterion(outputs['out'], label_imgs)
                train_loss += loss.item()

            # optimization step:
                self.optimizer.zero_grad() # (reset gradients)
                loss.backward() # (compute gradients)
                self.optimizer.step() # (perform optimization step)

            #the result presention/tensorboardX
                tensor = torch.tensor((), dtype=torch.float64) 
                y = tensor.new_zeros(outputs['out'].shape)
                #import pdb; pdb.set_trace()
                y[outputs['out'] > 0.5] = 1
                y[outputs['out'] < 0.5] = 0
                show_pred = torch.cat((y, y, y),1)
                show_gt = torch.cat((label_imgs, label_imgs, label_imgs),1)
                show_result = combine2tensor(show_pred, show_gt)
                #import pdb; pdb.set_trace()
                self.writer.add_images(tag='images', 
                                       img_tensor=imgs, 
                                       global_step=global_step)
                '''
                self.writer.add_images(tag = 'show_pred',
                                       img_tensor = show_pred,
                                       global_step = global_step)
                
                self.writer.add_images(tag = 'gt_img',
                                       img_tensor = show_gt,
                                       global_step = global_step)
                '''
                self.writer.add_images(tag = 'result',
                                        img_tensor = show_result,
                                        global_step = global_step)
                                                                
                self.writer.add_scalar(tag='Lr',
                                       scalar_value=self.optimizer.param_groups[0]['lr'],
                                       global_step=global_step)

                self.writer.add_scalars(main_tag='loss',
                                        tag_scalar_dict={'Cls Loss': train_loss / (batch_id + 1)},
                                        global_step=global_step)
                print('batch_id is %d'%(batch_id))
            print("each epoch use time : %f"%(time.time() - current_time))
            print ("####")

if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()

