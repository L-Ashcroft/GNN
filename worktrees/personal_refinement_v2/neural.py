"""Low-memory residual neural readout and AdamW; analytically checked gradients."""
import numpy as np


class Readout:
    def __init__(self, block, weights, intercept, rng, hidden=64, dropout=.05):
        n, k = weights.shape[1], weights.shape[0]
        bound = np.sqrt(6./(n+hidden))
        self.parameters = {'direct_w': weights.T.copy(), 'direct_b': intercept.copy(),
            'hidden_w': rng.uniform(-bound,bound,(n,hidden)), 'hidden_b': np.zeros(hidden),
            'output_w': np.zeros((hidden,k)), 'output_b': np.zeros(k)}
        self.first = {k:np.zeros_like(v) for k,v in self.parameters.items()}
        self.second = {k:np.zeros_like(v) for k,v in self.parameters.items()}
        self.step = 0; self.dropout = dropout
        self.slots = block['normalization']['slots']
        self.target = (block['train_z'][:,self.slots]-np.array(block['normalization']['target_center'])[self.slots]) / np.array(block['normalization']['target_std'])[self.slots]
        self.mask = block['train_mask'][:,self.slots]

    def forward(self, x, rng=None):
        p=self.parameters; h=x@p['hidden_w']+p['hidden_b']
        sigmoid=np.exp(-np.logaddexp(0.,-h)); active=h*sigmoid
        keep=np.ones_like(active)
        if rng is not None and self.dropout:
            keep=(rng.random(active.shape)>=self.dropout)/(1.-self.dropout)
        dropped=active*keep
        pred=x@p['direct_w']+p['direct_b']+dropped@p['output_w']+p['output_b']
        return pred,(x,h,sigmoid,dropped,keep)

    def backward(self, x, indices, rng=None):
        pred,cache=self.forward(x,rng); target=self.target[indices]; mask=self.mask[indices]
        count=int(mask.sum()); assert count>0
        error=np.where(mask,pred-target,0.); loss=float(np.square(error).sum()/count)
        delta=2.*error/count
        x,h,sigmoid,dropped,keep=cache
        hidden=(delta@self.parameters['output_w'].T)*keep*(sigmoid+h*sigmoid*(1.-sigmoid))
        grad={'direct_w':x.T@delta,'direct_b':delta.sum(0),'output_w':dropped.T@delta,
              'output_b':delta.sum(0),'hidden_w':x.T@hidden,'hidden_b':hidden.sum(0)}
        assert np.isfinite(loss) and all(np.isfinite(v).all() for v in grad.values())
        return loss,grad

    def update(self, gradient, direct_lr, residual_lr, multiplier=1., weight_decay=.01):
        self.step+=1
        for name,p in self.parameters.items():
            g=gradient[name]*multiplier
            self.first[name]*=.9; self.first[name]+=.1*g
            self.second[name]*=.999;self.second[name]+=.001*np.square(g)
            lr=direct_lr if name.startswith('direct_') else residual_lr
            p*=1.-lr*weight_decay
            p-=lr*(self.first[name]/(1.-.9**self.step))/(np.sqrt(self.second[name]/(1.-.999**self.step))+1e-8)
            assert np.isfinite(p).all()

    def state(self):
        return {prefix+'__'+name:value for prefix,values in [('p',self.parameters),('m',self.first),('v',self.second)] for name,value in values.items()} | {'step':np.array(self.step)}

    def restore(self, state):
        for prefix,values in [('p',self.parameters),('m',self.first),('v',self.second)]:
            for name in values: values[name][...]=state[prefix+'__'+name]
        self.step=int(state['step'])
