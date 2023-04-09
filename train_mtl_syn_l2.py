import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torchvision.transforms as T
from torchvision import transforms,models
import torch.nn.functional as F
from warmup_scheduler import GradualWarmupScheduler

from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder 
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch.cuda.amp import autocast, GradScaler
import timm
from PIL import Image
import torchvision.transforms as T
import copy


batch_size = 64
epochs = 200
lr = 1e-3


def concat_batches(list_iters):
    domain_img_list = torch.tensor([]).cuda()
    domain_label_list = torch.tensor([]).cuda()
    # domain_img_list = torch.tensor([])
    for source in list_iters:
        src1_img_real, src1_label_real = source.next()
        while (src1_img_real is None):
            src1_img_real, src1_label_real = source.next()

        src1_img_real = src1_img_real.cuda()
        src1_label_real = src1_label_real.cuda()
        domain_img_list = torch.cat([domain_img_list, src1_img_real], dim=0)
        domain_label_list = torch.cat([domain_label_list, src1_label_real], dim=0)
    batchindex = list(range(len(domain_label_list)))
    random.shuffle(batchindex)
    return domain_img_list[batchindex, :], domain_label_list[batchindex].type(torch.LongTensor).cuda()

def compute_center_loss(features, centers, targets):
    features = features.view(features.size(0), -1)
    target_centers = centers[targets]
    criterion = torch.nn.MSELoss()
    center_loss = criterion(features, target_centers)
    return center_loss

def get_center_delta(features, centers, targets, alpha):
    # implementation equation (4) in the center-loss paper
    features = features.view(features.size(0), -1)
    targets, indices = torch.sort(targets)
    target_centers = centers[targets]
    features = features[indices]

    delta_centers = target_centers - features
    uni_targets, indices = torch.unique(
            targets.cpu(), sorted=True, return_inverse=True)

    uni_targets = uni_targets.to(device)
    indices = indices.to(device)

    delta_centers = torch.zeros(
        uni_targets.size(0), delta_centers.size(1)
    ).to(device).index_add_(0, indices, delta_centers)

    targets_repeat_num = uni_targets.size()[0]
    uni_targets_repeat_num = targets.size()[0]
    targets_repeat = targets.repeat(
            targets_repeat_num).view(targets_repeat_num, -1)
    uni_targets_repeat = uni_targets.unsqueeze(1).repeat(
            1, uni_targets_repeat_num)
    same_class_feature_count = torch.sum(
            targets_repeat == uni_targets_repeat, dim=1).float().unsqueeze(1)

    delta_centers = delta_centers / (same_class_feature_count + 1.0) * alpha
    result = torch.zeros_like(centers)
    result[uni_targets, :] = delta_centers
    return result
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(2023)

class multi_modal(torch.nn.Module):
    def __init__(self):
        super(multi_modal, self).__init__()

        self.resnet_model = timm.create_model('resnet50', pretrained=True)#.to(device)
        num_ftrs = self.resnet_model.fc.in_features
        self.resnet_model.fc = nn.Linear(num_ftrs, 512)

        self.x1 =  nn.Linear(512,256)
        nn.init.xavier_normal_(self.x1.weight)
        self.bn1 = nn.BatchNorm1d(256,eps = 2e-1)
        #heads
        self.y1o = nn.Linear(256,2)
        nn.init.xavier_normal_(self.y1o.weight)#
        self.y2o = nn.Linear(256,2)
        nn.init.xavier_normal_(self.y2o.weight)

        #center inform
        self.centers = (torch.rand(2, 256).to(device) - 0.5) * 2

    def forward(self, x, is_align):
        feat = self.resnet_model(x)
        feat =  self.bn1(F.relu(self.x1(feat)))
        if is_align:
            # y = F.softmax(self.y1o(feat),dim=1)
            y = self.y1o(feat)
        else :
            # y = F.softmax(self.y2o(feat),dim=1)
            y = self.y2o(feat)
            
        feat = feat.div(
            torch.norm(feat, p=2, dim=1, keepdim=True).expand_as(feat))
        return y, feat


class CustumDataset(Dataset):
    def __init__(self, x, y, mode):
        self.photo_path = x
        self.photo_label = y
        if mode=='crop':
            self.transform = transforms.Compose([
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
                                        ])
        elif mode=='patch':
            self.transform = transforms.Compose([
                                        transforms.RandomCrop((130,130)),
                                        transforms.Resize((256,256)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
                                        ])

        else:
            self.transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
                               ])
                           

    def __len__(self):
        return len(self.photo_path)

    def __getitem__(self, item):
        img_path = self.photo_path[item]
        label = self.photo_label[item]
        img = Image.open(img_path)
        img = self.transform(img)
        return img, label



##################dataset for patch-based cropping#####################
#you shold modify this part
# idiap_train_dir_r = '/mnt/tmp/Idiap_frames/train/real'#'/mnt/tmp/Idiap_cropped_train/real'
# idiap_train_dir_a = '/mnt/tmp/Idiap_frames/train/attack'#'/mnt/tmp/Idiap_cropped_train/attack'
# idiap_train_dir2_r = '/mnt/tmp/idiap_whole_train/real'
# idiap_train_dir2_a = '/mnt/tmp/idiap_whole_train/attack'
msu_train_dir_r = '/mnt/tmp/MSU_cropped_v1/train/real'
msu_train_dir_a = '/mnt/tmp/MSU_cropped_v1/train/attack'
msu_train_dir2_r = '/mnt/tmp/msu_whole_train/real'
msu_train_dir2_a = '/mnt/tmp/msu_whole_train/attack'
oulu_train_dir_r = '/mnt/tmp/OULU_ImageFolder_v1/train/real/'#'/mnt/tmp/OULU_cropped/train/real'
oulu_train_dir_a = '/mnt/tmp/OULU_ImageFolder_v1/train/attack/'#'/mnt/tmp/OULU_cropped/train/attack'
oulu_train_dir2_r = '/mnt/tmp/oulu_whole_train/real'
oulu_train_dir2_a = '/mnt/tmp/oulu_whole_train/attack'
ca_train_dir_r = '/mnt/tmp/CASIA_cropped_v2/train/real'
ca_train_dir_a = '/mnt/tmp/CASIA_cropped_v2/train/attack'
ca_train_dir2_r = '/mnt/tmp/casia_whole_train/real'
ca_train_dir2_a = '/mnt/tmp/casia_whole_train/attack'

#you should annot this part if you don't use dataset on the protocal you will train
#idiap
# idiap_train_dir_r = [os.path.join(idiap_train_dir_r ,n) for n in os.listdir(idiap_train_dir_r)]
# # idiap_train_dir_a = [os.path.join(idiap_train_dir_a ,n) for n in os.listdir(idiap_train_dir_a)]
# idiap_train_dir_a = [os.path.join(idiap_train_dir_a+'/fixed' ,n) for n in os.listdir(idiap_train_dir_a+'/fixed')]\
#                     + [os.path.join(idiap_train_dir_a+'/hand' ,n) for n in os.listdir(idiap_train_dir_a+'/hand')]
# idiap_train_dir2_r = [os.path.join(idiap_train_dir2_r ,n) for n in os.listdir(idiap_train_dir2_r)]
# # idiap_train_dir2_a = [os.path.join(idiap_train_dir2_a ,n) for n in os.listdir(idiap_train_dir2_a)]
# idiap_train_dir2_a = [os.path.join(idiap_train_dir2_a+'/fixed' ,n) for n in os.listdir(idiap_train_dir2_a+'/fixed')]\
#                     + [os.path.join(idiap_train_dir2_a+'/hand' ,n) for n in os.listdir(idiap_train_dir2_a+'/hand')]
#msu
msu_train_dir_r = [os.path.join(msu_train_dir_r ,n) for n in os.listdir(msu_train_dir_r)]
msu_train_dir_a = [os.path.join(msu_train_dir_a ,n) for n in os.listdir(msu_train_dir_a)]
msu_train_dir2_r = [os.path.join(msu_train_dir2_r ,n) for n in os.listdir(msu_train_dir2_r)]
msu_train_dir2_a = [os.path.join(msu_train_dir2_a ,n) for n in os.listdir(msu_train_dir2_a)]
#oulu
oulu_train_dir_r = [os.path.join(oulu_train_dir_r ,n) for n in os.listdir(oulu_train_dir_r)]
oulu_train_dir_a = [os.path.join(oulu_train_dir_a ,n) for n in os.listdir(oulu_train_dir_a)]
oulu_train_dir2_r = [os.path.join(oulu_train_dir2_r ,n) for n in os.listdir(oulu_train_dir2_r)]
oulu_train_dir2_a = [os.path.join(oulu_train_dir2_a ,n) for n in os.listdir(oulu_train_dir2_a)]
#casia
ca_train_dir_r = [os.path.join(ca_train_dir_r ,n) for n in os.listdir(ca_train_dir_r)]
ca_train_dir_a = [os.path.join(ca_train_dir_a ,n) for n in os.listdir(ca_train_dir_a)]
ca_train_dir2_r = [os.path.join(ca_train_dir2_r ,n) for n in os.listdir(ca_train_dir2_r)]
ca_train_dir2_a = [os.path.join(ca_train_dir2_a ,n) for n in os.listdir(ca_train_dir2_a)]

oulu_test_dir = '/mnt/tmp/OULU_cropped/val' #crop
idiap_test_dir = '/mnt/tmp/Idiap_frames_v1/Idiap_frames/devel'
msu_test_dir = '/mnt/tmp/MSU_cropped_v1/train'#MSU crop``
# msu_test_dir = '/mnt/tmp/Msu_cropped_val'#MSU crop``
casia_test_dir = '/mnt/tmp/CASIA_cropped_val'#CASIA crop``

file_name = "resnet_M_O_C_MTL_whole_center_loader_clean_warmup_pat130_l1" #modify
full_name = f'/mnt/Face_Private-NFS/mhkim/swap/results_origin_clean/' #modify
os.makedirs(full_name, exist_ok=True)
full_name = full_name + file_name
print(file_name)


#you should annot this part if you don't use dataset on the protocal you will train
#idiap
# train_dataset_i_r = DataLoader(CustumDataset(idiap_train_dir_r, [1]*len(idiap_train_dir_r), mode='crop'), batch_size=10, shuffle=True, num_workers=8)
# train_dataset_i_a = DataLoader(CustumDataset(idiap_train_dir_a, [0]*len(idiap_train_dir_a), mode='crop'), batch_size=10, shuffle=True, num_workers=8)
# train_dataset_i2_r = DataLoader(CustumDataset(idiap_train_dir2_r, [1]*len(idiap_train_dir2_r), mode='patch'), batch_size=10, shuffle=True, num_workers=8)
# train_dataset_i2_a = DataLoader(CustumDataset(idiap_train_dir2_a, [0]*len(idiap_train_dir2_a), mode='patch'), batch_size=10, shuffle=True, num_workers=8)
#msu
train_dataset_m_r = DataLoader(CustumDataset(msu_train_dir_r, [1]*len(msu_train_dir_r), mode='crop'), batch_size=11, shuffle=True, num_workers=8)
train_dataset_m_a = DataLoader(CustumDataset(msu_train_dir_a, [0]*len(msu_train_dir_a), mode='crop'), batch_size=11, shuffle=True, num_workers=8)
train_dataset_m2_r = DataLoader(CustumDataset(msu_train_dir2_r, [1]*len(msu_train_dir2_r), mode='patch'), batch_size=11, shuffle=True, num_workers=8)
train_dataset_m2_a = DataLoader(CustumDataset(msu_train_dir2_a, [0]*len(msu_train_dir2_a), mode='patch'), batch_size=11, shuffle=True, num_workers=8)
#oulu
train_dataset_o_r = DataLoader(CustumDataset(oulu_train_dir_r, [1]*len(oulu_train_dir_r), mode='crop'), batch_size=11, shuffle=True, num_workers=8)
train_dataset_o_a = DataLoader(CustumDataset(oulu_train_dir_a, [0]*len(oulu_train_dir_a), mode='crop'), batch_size=11, shuffle=True, num_workers=8)
train_dataset_o2_r = DataLoader(CustumDataset(oulu_train_dir2_r, [1]*len(oulu_train_dir2_r), mode='patch'), batch_size=11, shuffle=True, num_workers=8)
train_dataset_o2_a = DataLoader(CustumDataset(oulu_train_dir2_a, [0]*len(oulu_train_dir2_a), mode='patch'), batch_size=11, shuffle=True, num_workers=8)
#casia
train_dataset_c_r = DataLoader(CustumDataset(ca_train_dir_r, [1]*len(ca_train_dir_r), mode='crop'), batch_size=11, shuffle=True, num_workers=8)
train_dataset_c_a = DataLoader(CustumDataset(ca_train_dir_a, [0]*len(ca_train_dir_a), mode='crop'), batch_size=11, shuffle=True, num_workers=8)
train_dataset_c2_r = DataLoader(CustumDataset(ca_train_dir2_r, [1]*len(ca_train_dir2_r), mode='patch'), batch_size=11, shuffle=True, num_workers=8)
train_dataset_c2_a = DataLoader(CustumDataset(ca_train_dir2_a, [0]*len(ca_train_dir2_a), mode='patch'), batch_size=11, shuffle=True, num_workers=8)

# src1_train_iter_real = iter(train_dataset_i_r); src1_train_iter_attack = iter(train_dataset_i_a)
# src1_iter_per_epoch_real = len(train_dataset_i_r); src1_iter_per_epoch_attack = len(train_dataset_i_a)
# src2_train_iter_real = iter(train_dataset_i2_r); src2_train_iter_attack = iter(train_dataset_i2_a)
# src2_iter_per_epoch_real = len(train_dataset_i2_r); src2_iter_per_epoch_attack = len(train_dataset_i2_a)
src1_train_iter_real = iter(train_dataset_m_r); src1_train_iter_attack = iter(train_dataset_m_a)
src1_iter_per_epoch_real = len(train_dataset_m_r); src1_iter_per_epoch_attack = len(train_dataset_m_a)
src2_train_iter_real = iter(train_dataset_m2_r); src2_train_iter_attack = iter(train_dataset_m2_a)
src2_iter_per_epoch_real = len(train_dataset_m2_r); src2_iter_per_epoch_attack = len(train_dataset_m2_a)
src3_train_iter_real = iter(train_dataset_o_r); src3_train_iter_attack = iter(train_dataset_o_a)
src3_iter_per_epoch_real = len(train_dataset_o_r); src3_iter_per_epoch_attack = len(train_dataset_o_a)
src4_train_iter_real = iter(train_dataset_o2_r); src4_train_iter_attack = iter(train_dataset_o2_a)
src4_iter_per_epoch_real = len(train_dataset_o2_r); src4_iter_per_epoch_attack = len(train_dataset_o2_a)
src5_train_iter_real = iter(train_dataset_c_r); src5_train_iter_attack = iter(train_dataset_c_a)
src5_iter_per_epoch_real = len(train_dataset_c_r); src5_iter_per_epoch_attack = len(train_dataset_c_a)
src6_train_iter_real = iter(train_dataset_c2_r); src6_train_iter_attack = iter(train_dataset_c2_a)
src6_iter_per_epoch_real = len(train_dataset_c2_r); src6_iter_per_epoch_attack = len(train_dataset_c2_a)
    
#testset loading
test_transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
                    ])

#you can modify this line as test dataset
test_dataset_i = ImageFolder(idiap_test_dir, transform= test_transform)
test_loader = DataLoader(test_dataset_i, batch_size=128, shuffle=False,num_workers=10)

#loss
device = 'cuda'
criterion = [nn.CrossEntropyLoss()]*2#.to(device)
mse = nn.MSELoss()

model = multi_modal().to(device)
optimizer = optim.SGD(
    [
        {"params":model.resnet_model.parameters(),"lr": lr},
        {"params":model.x1.parameters(), "lr": lr},
        {"params":model.y1o.parameters(), "lr": lr},
        {"params":model.y2o.parameters(), "lr": lr},
    ])
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[70,100,130,170,200], gamma=0.3)
scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=scheduler)

scaler = GradScaler()

best_acer = 1000
best_acer_test = 1000
result_list = []
label_list = []
predicted_list= []
start_epoch = 0 
if not os.path.exists(f'{full_name}.txt'):
    f = open(f'{full_name}.txt', "w")

# pretraining load
if os.path.exists(f'{full_name}_current.pth'):
    checkpoint = torch.load(f'{full_name}_current.pth')
    start_epoch = checkpoint['epoch']
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.load_state_dict(checkpoint['state_dict'])
    best_acer = checkpoint['best_acer']
    scheduler.load_state_dict(checkpoint['scheduler'])
    print("success loading pre-trained model ! ")
    
print('start_epoch : ',start_epoch)
for epoch in range(start_epoch, epochs):
    epoch_loss = 0
    epoch_accuracy = 0
    item_acc, item_acc_v2, item_loss, item_cnt = 0,0,0,0
    item_acc2 = 0
    model.train()
    max_iter = 2000
    scheduler_warmup.step()

    for iter_num in tqdm(range(max_iter+1)):
        # for data imbalance, we implement for dividing same num of data for each domain
        # you should annot this part if you don't use as training dataset

        # if (iter_num % src1_iter_per_epoch_real == 0):
        #     src1_train_iter_real = iter(train_dataset_i_r)
        # if (iter_num % src2_iter_per_epoch_real == 0):
        #     src2_train_iter_real = iter(train_dataset_i2_r)
        if (iter_num % src1_iter_per_epoch_real == 0):
            src1_train_iter_real = iter(train_dataset_m_r)
        if (iter_num % src2_iter_per_epoch_real == 0):
            src2_train_iter_real = iter(train_dataset_m2_r)
        if (iter_num % src3_iter_per_epoch_real == 0):
            src3_train_iter_real = iter(train_dataset_o_r)
        if (iter_num % src4_iter_per_epoch_real == 0):
            src4_train_iter_real = iter(train_dataset_o2_r)
        if (iter_num % src5_iter_per_epoch_real == 0):
            src5_train_iter_real = iter(train_dataset_c_r)
        if (iter_num % src6_iter_per_epoch_real == 0):
            src6_train_iter_real = iter(train_dataset_c2_r)

        # if (iter_num % src1_iter_per_epoch_attack== 0):
        #     src1_train_iter_attack = iter(train_dataset_i_a)
        # if (iter_num % src2_iter_per_epoch_attack == 0):
        #     src2_train_iter_attack = iter(train_dataset_i2_a)
        if (iter_num % src1_iter_per_epoch_attack == 0):
            src1_train_iter_attack = iter(train_dataset_m_a)
        if (iter_num % src2_iter_per_epoch_attack == 0):
            src2_train_iter_attack = iter(train_dataset_m2_a)
        if (iter_num % src3_iter_per_epoch_attack == 0):
            src3_train_iter_attack = iter(train_dataset_o_a)
        if (iter_num % src4_iter_per_epoch_attack == 0):
            src4_train_iter_attack = iter(train_dataset_o2_a)
        if (iter_num % src5_iter_per_epoch_attack == 0):
            src5_train_iter_attack = iter(train_dataset_c_a)
        if (iter_num % src6_iter_per_epoch_attack == 0):
            src6_train_iter_attack = iter(train_dataset_c2_a)

        list_iters= [src1_train_iter_real, src1_train_iter_attack,\
                    src3_train_iter_real, src3_train_iter_attack,\
                    src5_train_iter_real, src5_train_iter_attack,\
                    ]
        list_iters_align= [src2_train_iter_real, src2_train_iter_attack,\
                            src4_train_iter_real, src4_train_iter_attack,\
                            src6_train_iter_real, src6_train_iter_attack,\
                            ]
        data, label = concat_batches(list_iters)
        data_align, target_align = concat_batches(list_iters_align)
        
        data_align = data_align.to(device)
        target_align = target_align.to(device)
        data = data.to(device)
        label = label.to(device)
        centers = model.centers

        if len(target_align) != len(label):
            continue

        with autocast(enabled=True):
            output_a, feat_a = model(data_align, is_align=True)
            loss0 = criterion[0](output_a, target_align)
            output, feat = model(data, is_align=False)
            loss1 = criterion[1](output, label)
            loss_dist = 0

            center_loss = compute_center_loss(feat, centers, label)
            l2_norm = sum(p.pow(2.0).sum()
                        for p in model.parameters())
            loss = loss0 + loss1 + 0.2*center_loss + 0.01 * l2_norm

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        center_deltas = get_center_delta(
            feat.data, centers, label,alpha=0.5)
        model.centers = centers - center_deltas

        item_acc += ((output_a + output).argmax(dim=1) == label).float().sum().item()
        item_acc_v2+= ((output).argmax(dim=1) == label).float().sum().item()
        item_loss += torch.mean(loss).item()
        item_cnt += 1
        
    epoch_accuracy_1 = item_acc / item_cnt
    epoch_accuracy_2 = item_acc_v2 / item_cnt
    epoch_loss = item_loss / item_cnt
    
    model.eval()
    with torch.no_grad():
        epoch_val_accuracy = 0
        epoch_val_loss = 0
        acer_origin =0 
            
        #saving according to test domain (validation set)
        result_list = []
        label_list = []
        predicted_list= []
        _epoch_val_accuracy = 0
        _epoch_val_loss = 0
        _item_acc, _item_loss = 0,0
        epoch_val_accuacy = 0
        epoch_val_loss = 0
        acer_testset =0 

        for data, label in test_loader:
            data = data.cuda()
            label = label.cuda()

            val_output, _ = model(data, is_align=False)
            loss = criterion[1](val_output, label)
            _,predicted = torch.max((val_output).data, 1)
            predicted = predicted.to('cpu').detach().numpy()
            _item_acc += ((val_output).argmax(dim=1) == label).float().sum().item()
            _item_loss += torch.mean(loss).item()
            label = label.to('cpu').detach().numpy()
            val_o = ((val_output).to('cpu').detach().numpy())

            for i_batch in range(val_o.shape[0]):
                    result_list.append(val_o[i_batch,1])
                    label_list.append(label[i_batch])
                    predicted_list.append(predicted[i_batch])
            # break
        _epoch_val_accuracy = _item_acc / (len(test_loader.dataset))
        _epoch_val_loss = _item_loss / (len(test_loader.dataset))
        tn, fp, fn, tp = confusion_matrix(label_list, predicted_list, labels=[0,1]).ravel()
        f = open(full_name+'.txt', "a")
        f.write(f'tn, fp, fn, tp :,{tn}, {fp}, {fn}, {tp}\n')
        print('tn, fp, fn, tp : ',tn, fp, fn, tp )
        f.close()


        npcer = fp/(tn + fp) if (tn + fp) != 0 else 1000
        apcer = fn/(fn + tp) if (fn + tp) != 0 else 1000
        _acer = (apcer + npcer)/2

        epoch_val_accuracy += _epoch_val_accuracy
        epoch_val_loss += _epoch_val_loss
        acer_testset += _acer

        f = open(full_name+'.txt', "a")
        print(f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - apcer | npcer : {apcer:.4f} | {npcer:.4f} - acer : {_acer:.4f} - acc: {epoch_accuracy_1:.4f}, acc2:{epoch_accuracy_2:.4f} - loss: {epoch_loss:.4f} - val_acc: {_epoch_val_accuracy:.4f} - val_loss : {_epoch_val_loss:.4f}")
        f.write(f"CASIA - Epoch : {epoch+1} - loss : {epoch_loss:.4f} - apcer | npcer : {apcer:.4f} | {npcer:.4f} - acer : {_acer:.4f} - acc: {epoch_accuracy_1:.4f}, acc2:{epoch_accuracy_2:.4f} - loss: {epoch_loss:.4f} - val_acc: {_epoch_val_accuracy:.4f} - val_loss : {_epoch_val_loss:.4f}\n")
        f.close()

    dict_satates = {\
    'epoch': epoch + 1,\
    'state_dict': model.state_dict(),\
    'best_acer': best_acer, \
    'optimizer': optimizer.state_dict(),\
    'scheduler': scheduler.state_dict(),\
    'scheduler_warmup': scheduler_warmup.state_dict()
    }

    if acer_testset < best_acer_test:
        best_acer_test = acer_testset
        dict_satates['best_acer'] = acer_testset
        torch.save(dict_satates, f'{full_name}_testset.pth') ############ change model name !!
        if epoch < 70:
            torch.save(dict_satates, f'{full_name}_testset_until70.pth') ############ change model name !!
        f = open(full_name+'.txt', "a")
        f.write(f"current best acer : {acer_testset}")
        print(f"current best acer : {acer_testset}")
        f.close()

    torch.save(dict_satates, f'{full_name}_current.pth')

        
