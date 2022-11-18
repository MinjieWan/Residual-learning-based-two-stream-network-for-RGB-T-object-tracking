from collections import OrderedDict

opts = OrderedDict()
opts['use_gpu'] = True # if you change to False ,it will use CPU to run,but is very slow
opts['model_path1'] = "/home/test/chenyili/MANet-master-change/models/final33.pth"

opts['img_size'] = 107
opts['padding'] = 16

opts['batch_pos'] = 24
opts['batch_neg'] = 72
opts['batch_neg_cand'] = 768
opts['batch_test'] = 192

opts['n_samples'] = 512

opts['trans_f'] = 0.6
opts['scale_f'] = 1.05
opts['trans_f_expand'] = 1.5 

opts['n_bbreg'] = 1000
opts['overlap_bbreg'] = [0.6, 1]
opts['scale_bbreg'] = [1,2]

opts['lr_init'] = 0.001
opts['maxiter_init'] = 30 
opts['n_pos_init'] = 500
opts['n_neg_init'] = 5000
opts['overlap_pos_init'] = [0.7, 1]
opts['overlap_neg_init'] = [0, 0.5]

opts['lr_update'] = 0.0002
opts['maxiter_update'] = 15
opts['n_pos_update'] = 50
opts['n_neg_update'] = 200
opts['overlap_pos_update'] = [0.7, 1]
opts['overlap_neg_update'] = [0, 0.3]

opts['success_thr'] = 0
opts['n_frames_short'] = 20
opts['n_frames_long'] = 100 #init 100
opts['long_interval'] = 5

opts['w_decay'] = 0.0005
opts['momentum'] = 0.9
opts['grad_clip'] =10
opts['lr_mult'] = {'fc6':10,'fc4':5,'fc5':5}   #test gtot use {'fc6':10,'fc4':5,'fc5':5}  ///  #test rgbt234 use {'fc6':10}
opts['ft_layers'] = ['fc']
