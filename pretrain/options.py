from collections import OrderedDict

opts = OrderedDict()
opts['use_gpu'] = True

opts['init_model_path'] = ''

opts['model_path'] = '/home/test/chenyili/MANet-master-change/models/final33.pth'

opts['batch_frames'] = 8
opts['batch_pos'] = 32
opts['batch_neg'] = 96

opts['overlap_pos'] = [0.7, 1]
opts['overlap_neg'] = [0, 0.5]

opts['img_size'] = 107
opts['padding'] = 16

opts['lr'] = 0.001
opts['w_decay'] = 0.0005
opts['momentum'] = 0.9
opts['grad_clip'] = 10
opts['ft_layers'] = ['fc','R','T','conv']
opts['lr_mult'] = {'fc':2,'R':1,'T':1,'conv':1}

opts['n_cycles'] = 50
