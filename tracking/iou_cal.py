import sys
import os
sys.path.append('/home/a11134315/MANet-master/tracking')
import numpy as np
from utils import overlap_ratio
gt_path='/home/a11134315/RGB_T234/manwithbasketball/init.txt'
# predict_path='/home/a11134315/py-MDNet-master/results/floor-1/result.txt'
predict_path='/home/a11134315/MANet-master/MANet_RGBT234/MANet_manwithbasketball.txt'
# predict_path='/home/a11134315/rgbt-tracking-fsrpn-master/results/manwithbasketball/result.txt'
result='/home/a11134315/iou_results'
result_path=os.path.join(result,'manwithbasketball.txt')
# if not os.path.exists(result_path):
#     os.makedirs(result_path)
gt = np.loadtxt(gt_path,delimiter=',')
try:
    predict=np.loadtxt(predict_path,delimiter=',')
except:
    predict = np.loadtxt(predict_path, delimiter=' ')
if predict.shape[1] == 8:
    x_min = np.min(predict[:, [0, 2, 4, 6]], axis=1)[:, None]
    y_min = np.min(predict[:, [1, 3, 5, 7]], axis=1)[:, None]
    x_max = np.max(predict[:, [0, 2, 4, 6]], axis=1)[:, None]
    y_max = np.max(predict[:, [1, 3, 5, 7]], axis=1)[:, None]
    predict = np.concatenate((x_min, y_min, x_max - x_min, y_max - y_min), axis=1)
iou=np.zeros(len(gt))
f=open(result_path,'w')
for i in range(len(gt)):
    iou[i]=overlap_ratio(gt[i],predict[i])
    f.write(str(iou[i]))
    f.write('\n')
a=np.mean(iou)
f.write('mean iou=')
f.write(str(a))
f.close()
