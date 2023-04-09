import os
import random
# import timm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torchvision.transforms as T
from torchvision import transforms,models
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder 
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch.cuda.amp import autocast, GradScaler
import timm

batch_size = 64
epochs = 70
lr = 1e-3

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

    def forward(self, x):
        x = self.resnet_model(x)
        x =  self.bn1(F.relu(self.x1(x)))
        y1o = F.softmax(self.y1o(x),dim=1)
        y2o = F.softmax(self.y2o(x),dim=1)

        return y1o, y2o

#change dataset path !
oulu_test_dir = '/mnt/tmp/OULU_ImageFolder_v1/test'
msu_test_dir = '/mnt/tmp/MSU_cropped_v1/test'
idiap_test_dir = '/mnt/tmp/Idiap_frames_v1/test/'
casia_test_dir = '/mnt/tmp/CASIA_cropped_v2/test'
list_name = []
#change weight path name you will infer (without '.pth')
list_name.append("results_origin_clean/resnet_O_I_C_SYN_MTL_whole_center_loader_clean_warmup_pat130_l1")
for file_name in list_name:
    full_name = f'/mnt/Face_Private-NFS/mhkim/swap/{file_name}'
    transform_patch = transforms.Compose([
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
                                transforms.RandomCrop((50,50)),
                                transforms.Resize((256,256)),
                                ])

    test_transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
                                ])
                           
    # you shold annot if you don't use as test dataset
    # test_dataset_o = ImageFolder(oulu_test_dir, transform= test_transform)
    # test_loader_o = DataLoader(test_dataset_o, batch_size=batch_size, shuffle=False,num_workers=6)
    test_dataset_m = ImageFolder(msu_test_dir, transform= test_transform)
    test_loader_m = DataLoader(test_dataset_m, batch_size=batch_size, shuffle=False,num_workers=6)
    # test_dataset_i = ImageFolder(idiap_test_dir, transform= test_transform)
    # test_loader_i = DataLoader(test_dataset_i, batch_size=batch_size, shuffle=False,num_workers=6)
    # test_dataset_c = ImageFolder(casia_test_dir, transform= test_transform)
    # test_loader_c = DataLoader(test_dataset_c, batch_size=batch_size, shuffle=False,num_workers=6)

    #change test dataset variance
    list_test_loader = [('msu',test_loader_m)]#

    # loss function
    device = 'cuda'
    criterion = [nn.CrossEntropyLoss()]*2
    model = multi_modal().to(device)
    optimizer = optim.SGD(
        [
            {"params":model.resnet_model.parameters(),"lr": lr},
            {"params":model.x1.parameters(), "lr": lr},
            {"params":model.y1o.parameters(), "lr": lr},
            {"params":model.y2o.parameters(), "lr": lr},
        ])
    scaler = GradScaler()

    best_acer = 1000
    result_list = []
    label_list = []
    predicted_list= []
    if not os.path.exists(f'{full_name}.txt'):
        f = open(f'{full_name}.txt', "w")
        f.close()

    # pretraining load
    print('current_epoch : ',torch.load(f'{full_name}_current.pth')['epoch'])
    print(full_name)
    if os.path.exists(f'{full_name}_testset.pth'):
        checkpoint = torch.load(f'{full_name}_testset.pth')
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['state_dict'])
        best_acer = checkpoint['best_acer']
        print("success loading pre-trained model ! ")
    else:
        start_epoch = 0 
        beak
    print('start_epoch : ',start_epoch)
    import torchvision.transforms as T
    for epoch in range(1):
        epoch_loss = 0
        epoch_accuracy = 0
        item_acc, item_loss, item_cnt = 0,0,0
        item_acc2 = 0
        model.eval()
        with torch.no_grad():
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            acer =0 

            for name, test_loader in list_test_loader:
                f = open(f'{full_name}.txt', "a")
                f.write(f"{name},\n")
                print(f"{name},\n")
                f.close()

                result_list = []
                label_list = []
                predicted_list= []
                _epoch_val_accuracy = 0
                _epoch_val_loss = 0
                _item_acc, _item_loss = 0,0

                for data, label in test_loader:
                    data = data.cuda()
                    label = label.cuda()

                    val_output = model(data)
                    loss0 = criterion[0](val_output[0], label)
                    loss1 = criterion[1](val_output[1], label)
                    loss = loss0 + loss1
                
                    _,predicted = torch.max((val_output[0]+val_output[1]).data, 1)
                    predicted = predicted.to('cpu').detach().numpy()
                    _item_acc += ((val_output[0]+val_output[1]).argmax(dim=1) == label).float().sum().item()
                    _item_loss += torch.mean(loss).item()
                    label = label.to('cpu').detach().numpy()
                    val_o = ((val_output[0]+val_output[1]).to('cpu').detach().numpy())/2.

                    for i_batch in range(val_o.shape[0]):
                            result_list.append(val_o[i_batch,1])
                            label_list.append(label[i_batch])
                            predicted_list.append(predicted[i_batch])
                    
                _epoch_val_accuracy = _item_acc / (len(test_loader.dataset))
                _epoch_val_loss = _item_loss / (len(test_loader.dataset))
                tn, fp, fn, tp = confusion_matrix(label_list, predicted_list, labels=[0,1]).ravel()
                auc_score = roc_auc_score(label_list, result_list) * 100
                print(tn, fp, fn, tp )
                npcer = fp/(tn + fp) if (tn + fp) != 0 else 1000
                apcer = fn/(fn + tp) if (fn + tp) != 0 else 1000
                _acer = (apcer + npcer)/2

                epoch_val_accuracy += _epoch_val_accuracy/len(list_test_loader)
                epoch_val_loss += _epoch_val_loss/len(list_test_loader)
                acer += _acer/len(list_test_loader)

                f = open(full_name+'.txt', "a")
                print(f"Epoch : {epoch+1} - apcer | npcer : {apcer:.4f} | {npcer:.4f} - acer : {_acer:.4f} - auc : {auc_score:.2f} - val_acc: {_epoch_val_accuracy:.4f} - val_loss : {_epoch_val_loss:.4f}")
                f.write(f"Epoch : {epoch+1} - apcer | npcer : {apcer:.4f} | {npcer:.4f} - acer : {_acer:.4f} - auc : {auc_score:.2f} - val_acc: {_epoch_val_accuracy:.4f} - val_loss : {_epoch_val_loss:.4f}\n")
                f.close()
