import os
import torch
from ..models import Timer


def __init__(self, args):
    self.args = args
    self.model_dict = {
        'TrmEncoder': TrmEncoder,
        'Timer': Timer,
    }
    if self.args.use_multi_gpu:
        self.model = self._build_model()
        self.device = torch.device('cuda:{}'.format(self.args.gpu))
    else:
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'Timer': Timer,
            'Timer-UTSD': Timer,
            'Timer-LOTSA': Timer,
            'Timer1': Timer,
        }
        if self.args.use_multi_gpu:
            self.model = self._build_model()
            self.device = torch.device('cuda:{}'.format(self.args.local_rank))
        else:
            self.device = self._acquire_device()
            print('device:', self.device)
            # self.model = self._build_model()
            # print(f'os.environ["CUDA_VISIBLE_DEVICES"]={os.environ["CUDA_VISIBLE_DEVICES"]}')
            # devices = ['cpu', 'cuda:1', 'cuda:2', 'cuda:3']
            # # 输出 cuda available devices
            # print('cuda available devices:', torch.cuda.device_count())
            # # 看看那个会有错误
            # for device in devices:
            #     try:
            #         self.device = torch.device(device)
            #         print('device:', self.device)
            #         self.model = self._build_model().to(device)
            #     except Exception as e:
            #         print(e)
            self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            # FIXME
            # os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
            # os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7'
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device
    
    # def _acquire_device(self):
    #     if self.args.use_gpu:
    #         os.environ["CUDA_VISIBLE_DEVICES"] = str(
    #             self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
    #         device = torch.device('cuda:{}'.format(self.args.gpu))
    #         print('Use GPU: cuda:{}'.format(self.args.gpu))
    #     else:
    #         device = torch.device('cpu')
    #         print('Use CPU')
    #     return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
