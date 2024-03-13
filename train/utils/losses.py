import torch
import torch.nn as nn

# 예측값과 실제값을 입력으로 받아 손실을 계산하는 함수
class losses:
    def __init__(self, type):
        self.type = type 

    
    def __getitem__(self):
        if type == 'bce':
            return nn.BCELoss()
        else:
            return None