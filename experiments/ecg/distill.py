import torch.nn as nn
import torch
import torch.nn.functional as F
from options import Options ##

class DistillKL(nn.Module):
    def __init__(self, opt):
        super(DistillKL, self).__init__()
        opt = Options().parse()
        self.T = opt.temperature
        self.alpha= opt.alpha

    def forward(self, y_s, y_t):
        #B, C, H, W = y_s.size()
        y_s=torch.from_numpy(y_s)
        y_t=torch.from_numpy(y_t)
        y_s=y_s.reshape(1,len(y_s))
        y_t=y_t.reshape(1,len(y_t))
        #print("y_s reshape: ", y_s.shape)
        #print("y_t reshape: ",y_t.shape) 
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = self.alpha*F.kl_div(p_s, p_t.detach(), reduction='sum') * (self.T**2) / y_s.shape[0]
        return loss

"""  
class Attention(nn.Module):
    def __init__(self, opt):
        super(Attention, self).__init__()
        opt = Options().parse()
        self.p = 2
        self.kd = DistillKL(opt)
        self.alpha = opt.alpha
        self.beta = opt.beta
        print(opt.alpha, opt.beta)

    def forward(self, o_s, o_t, g_s, g_t):
        loss = self.alpha * self.kd(o_s, o_t)
        loss += self.beta * sum([self.at_loss(f_s, f_t.detach()) for f_s, f_t in zip(g_s, g_t)])

        return loss

    def at_loss(self, f_s, f_t):
        return (self.at(f_s) - self.at(f_t)).pow(2).mean()

    def at(self, f):
        return F.normalize(f.pow(self.p).mean(1).view(f.size(0), -1))
"""