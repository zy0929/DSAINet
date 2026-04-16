import torch
import torch.nn as nn
import torch.nn.functional as F

class ShallowConvNet(nn.Module):
    def convBlock(self, inF, outF, dropoutP, kernalSize, *args, **kwargs):
        return nn.Sequential(
            nn.Dropout(p=dropoutP),
            Conv2dWithConstraint(inF, outF, kernalSize, bias=False, max_norm=2, *args, **kwargs),
            nn.BatchNorm2d(outF),
            nn.ELU(),
            nn.MaxPool2d((1, 3), stride=(1, 3))
        )

    def firstBlock(self, outF, dropoutP, kernalSize, nChan, *args, **kwargs):
        return nn.Sequential(
            Conv2dWithConstraint(1, outF, kernalSize, padding=0, max_norm=2, *args, **kwargs),
            Conv2dWithConstraint(40, 40, (nChan, 1), padding=0, bias=False, max_norm=2),
            nn.BatchNorm2d(outF),
        )

    def calculateOutSize(self, nChan, nTime):
        '''
        Calculate the output based on input size.
        model is from nn.Module and inputSize is a array.
        '''
        data = torch.rand(1, 1, nChan, nTime)
        block_one = self.firstLayer
        avg = self.avgpool
        dp = self.dp
        out = torch.log(block_one(data).pow(2))
        out = avg(out)
        out = dp(out)
        out = out.view(out.size()[0], -1)
        return out.size()

    def __init__(self, nChan, nTime, nClass=2, dropoutP=0.25, *args, **kwargs):
        super(ShallowConvNet, self).__init__()

        kernalSize = (1, 25)
        nFilt_FirstLayer = 40

        self.firstLayer = self.firstBlock(nFilt_FirstLayer, dropoutP, kernalSize, nChan)
        self.avgpool = nn.AvgPool2d((1, 75), stride=(1, 15))
        self.dp = nn.Dropout(p=dropoutP)
        self.fSize = self.calculateOutSize(nChan, nTime)
        self.lastLayer = nn.Linear(self.fSize[-1], nClass)

    def forward(self, x):
        x = self.firstLayer(x)
        x = torch.log(self.avgpool(x.pow(2)))
        x = self.dp(x)
        x = x.view(x.size()[0], -1)
        x = self.lastLayer(x)

        return x
    
class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)


class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)
