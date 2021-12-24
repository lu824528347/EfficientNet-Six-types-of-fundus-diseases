import os

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'
import sys
import numpy as np
current_dir = os.path.abspath(os.path.dirname(__file__))
import torch
import torch.nn as nn
import csv
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tools.eye_dataset import eyeDataset
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
from tools.common_tools import get_resnet_18, get_resnet_50
import torchvision.models as models
# from src.transfer_resnet50 import resnet_model
from PIL import Image
from sklearn.metrics import confusion_matrix,cohen_kappa_score,hamming_loss, jaccard_score,hinge_loss,f1_score
# from sklearn.metrics import jaccard_similarity_score
import matplotlib.pyplot as plt
import time
from tools.performance import kappa

ASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_transforms = {
    'test': transforms.Compose([
        transforms.Resize(size=380),
        transforms.CenterCrop(size=380),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}
dataset = '/public/home/lubin/Test9_efficientNet'
test_directory = os.path.join(dataset, 'test')
data = {
    'test': datasets.ImageFolder(root=test_directory, transform=image_transforms['test'])
}
batch_size = 32
num_classes = 6
test_data_size = len(data['test'])
test_data = DataLoader(data['test'], batch_size=batch_size, shuffle=True)
classes =('0','1','2','3','4','5')
idx_to_class = {v: k for k, v in data['test'].class_to_idx.items()}

path = '/public/home/lubin/Test9_efficientNet/test'

def str2int(a):
    b = []
    for i in a:
        b.append(int(i))
    return b
def get_filelist(dir):
    Filelist = []
    r1 = []
    r2 = []
    for home, dirs, files in os.walk(path):
        for filename in files:
            # 文件名列表，包含完整路径
            Filelist.append(os.path.join(home, filename))
            # # 文件名列表，只包含文件名
            # Filelist.append( filename)

            r1.append(home[-1])
            r2.append(filename)
    return Filelist, r1, r2

l1 = []
l2 = []
l3 = []
l4 = []
def predict(model, test_image_name):
    transform = image_transforms['test']
    Filelist, r1 ,r2= get_filelist(dir)

    print(len(Filelist))
    # model = torch.load('50.pth')

    model =model.to(device)
    timeall = 0
    for file_index,file in enumerate(Filelist):


        test_image_name = file
        # print(test_image_name)
        test_image = Image.open(test_image_name)



        test_image_tensor = transform(test_image).unsqueeze(0)
        img_ = test_image_tensor.to(device)

        start = time.time()
        out = model(img_)
        end = time.time()
        timed = end - start
        timeall = timeall + timed
        print(timed*1000)#时间

        # ps = torch.exp(out)
        # topk, topclass = ps.topk(1, dim=1)
        _, predicted = torch.max(out, 1)
        percentage = torch.nn.functional.softmax(out, dim=1)[0]
        temp = classes[predicted[0]]

        p = int(r1[file_index])
            # flag = int(temp == p)

            # _k = 1 if temp == p else 0

        # print("image_name :", test_image_name, "Prediction : ", temp,
        #           ", Score: ", percentage[predicted[0]].item(), "Real :", p)
        l1.append(r2[file_index])
        l2.append(temp)
        l3.append(percentage[predicted[0]].item())
        l4.append(p)

    print("avg time: ", timeall * 1000 / len(Filelist), " ms")
            # test_image.show()
    return l1, l2, l3, l4


# if __name__ == "__main__":

model = torch.load('efficientB4.pth',map_location='cuda:0')
l1, l2, l3, l4 = predict(model, '/public/home/lubin/Test9_efficientNet/test')
k = {'image_name': l1,
     'Prediction': l2,
     'Score': l3,
     'Real': l4}

data = pd.DataFrame(k)
data.to_csv('B4.csv', index=None)
# print('{}\n{}\n{}\n{}'.format(l1, l2, l3, l4))
Filelist,r1,r2 = get_filelist(dir)
y_true = str2int(r1)
y_pred = str2int(l2)
labels = ['0','1','2','3','4','5']

tick_marks = np.array(range(len(labels))) + 0.5
def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.binary):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


cm = confusion_matrix(y_true, y_pred)
np.set_printoptions(precision=2)
# cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
# print cm_normalized
plt.figure(figsize=(12, 8), dpi=120)

ind_array = np.arange(len(labels))
x, y = np.meshgrid(ind_array, ind_array)

for x_val, y_val in zip(x.flatten(), y.flatten()):
    c = cm[y_val][x_val]
    if c > 0.01:
        plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')
# offset the tick
plt.gca().set_xticks(tick_marks, minor=True)
plt.gca().set_yticks(tick_marks, minor=True)
plt.gca().xaxis.set_ticks_position('none')
plt.gca().yaxis.set_ticks_position('none')
plt.grid(True, which='minor', linestyle='-')
plt.gcf().subplots_adjust(bottom=0.15)

plot_confusion_matrix(cm, title='test confusion matrix')
# show confusion matrix
plt.savefig('confusion_matrix.png', format='png')
plt.show()
TP = np.diag(cm)
FN = cm.sum(axis=1)-np.diag(cm)
FP = cm.sum(axis=0)-np.diag(cm)
TN = cm.sum() - (FP+FN+TP)
# Sensitivity
TPR = TP/(TP+FN)
# Specificity
TNR = TN/(TN+FP)
# Precision
PPV = TP/(TP+FP)
cls_num = len(labels)
# print(cls_num)
kappa = cohen_kappa_score(y_true,y_pred)
ham_distance = hamming_loss(y_true,y_pred)

# jaccrd_score =  jaccard_similarity_score(y_true,y_pred )
# hinge_loss = hinge_loss(y_true,y_pred)

# a=kappa(y_true,y_pred,min_rating=None, max_rating=None)
print('kappa:{:.2%}, ham_distance:{:.2%}'.format(kappa,ham_distance))
print('micro-f1:{:.2%}'.format(f1_score(y_true,y_pred,labels=[0,1,2,3,4,5],average='micro')))
print('macro-f1:{:.2%}'.format(f1_score(y_true,y_pred,labels=[0,1,2,3,4,5],average='macro')))

for i in range(cls_num):
    # print('TPR::::{}'.format(TPR[i]))
    # print('TNR::::{}'.format(TNR[i]))
    # print('PPV::::{}'.format(PPV[i]))
    print('class:{:<10}, total num:{:<6}, correct num:{:<5}  Recall( Sensitivity): {:.2%} Specificity: {:.2%} Precision: {:.2%} f1-score: {:.2%}'.format(
        labels[i], np.sum(cm[i, :]), cm[i, i],
        TPR[i],
        TNR[i],
        PPV[i],
        (2*TPR[i]*PPV[i])/(TPR[i]+PPV[i])

            ))
