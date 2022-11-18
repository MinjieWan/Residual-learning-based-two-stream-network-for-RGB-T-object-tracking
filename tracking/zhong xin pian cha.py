import numpy as np
import os
import sys
import math
# sys.path.append('/home/a11134315/MANet-master/tracking')
gt_path='/home/a11134315/RGB_T234/floor-1/init.txt'
# predict_path='/home/a11134315/py-MDNet-master/results/bus6/result.txt'
# predict_path='/home/a11134315/MANet-master/MANet_RGBT234/MANet_basketball2.txt'
predict_path='/home/a11134315/rgbt-tracking-fsrpn-master/results/floor-1/result.txt'
result='/home/a11134315/precision'
result_path=os.path.join(result,'floor-1.txt')
try:
    gt=np.loadtxt(gt_path,delimiter=',')
except:
    gt=np.loadtxt(gt_path,delimiter=' ')
try:
    predict=np.loadtxt(predict_path,delimiter=',')
except:
    predict=np.loadtxt(predict_path,delimiter=' ')
if predict.shape[1] == 8:
    x_min = np.min(predict[:, [0, 2, 4, 6]], axis=1)[:, None]
    y_min = np.min(predict[:, [1, 3, 5, 7]], axis=1)[:, None]
    x_max = np.max(predict[:, [0, 2, 4, 6]], axis=1)[:, None]
    y_max = np.max(predict[:, [1, 3, 5, 7]], axis=1)[:, None]
    predict = np.concatenate((x_min, y_min, x_max - x_min, y_max - y_min), axis=1)
precision=np.zeros(len(gt))
f=open(result_path,'w')
n=0
for i in range(0,len(gt)):
    x1=gt[i][0]+gt[i][2]/2
    y1=gt[i][1]+gt[i][3]/2
    x2=predict[i][0]+predict[i][2]/2
    y2=predict[i][1]+predict[i][3]/2
    precision[i]=math.sqrt((x2-x1)**2+(y2-y1)**2)
    if precision[i]<=20:
        n+=1
    f.write(str(precision[i]))
    f.write('\n')
rate=n/float(len(gt))
f.write('precision=')
f.write(str(rate))
f.close()