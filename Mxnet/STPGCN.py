# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 22:30:55 2021


@author: Yiji Zhao

"""

import numpy as np
from mxnet import nd
from mxnet.gluon import nn

class STPGConv(nn.HybridBlock):
    def __init__(self, config):
        super(STPGConv, self).__init__()
        
        self.C = config.C
        self.d = config.d
        self.V = config.V
        self.t_size = config.t_size
        with self.name_scope():
            self.W = self.params.get('W', shape=(self.C, self.C*2))
            self.b = self.params.get('b', shape=(1, self.C*2))
            self.Ws = self.params.get('Ws', shape=(self.d, self.C*2))
            self.Wt = self.params.get('Wt', shape=(self.d, self.C*2))
            self.ln  = nn.LayerNorm() 

    def hybrid_forward(self, F, x, S, sape, tape, W, b, Ws, Wt):
        # x:B,t,V,C
        # S:B,V,tV
        # SE: V,d
        # TE: B,1,1,d
        
        # aggregation
        # B,t,V,C -> B,tV,C
        x = F.reshape(x,(-1,self.t_size*self.V,self.C))
        # B,(V,tV x tV,C) -> B,V,C
        x = F.batch_dot(S,x)
        x = F.dot(x,W)
        x = F.broadcast_add(x,b)
        
        # STPGAU
        # V,d x d,C -> V,C 
        SE = F.dot(sape,Ws)
        # B,1,1,d -> B,1,1,d -> B,1,d
        TE = F.reshape(tape,(-1,1,self.d))
        # B,1,d x d,C -> B,1,C 
        TE = F.dot(TE,Wt)
        x = F.broadcast_add(x,SE)
        x = F.broadcast_add(x,TE)
        x = self.ln(x)
        lhs, rhs = F.split(x, num_outputs=2, axis=-1)
        x = lhs * F.sigmoid(rhs)
        return x


class Gaussian_component(nn.HybridBlock):
    def __init__(self, config):
        super(Gaussian_component, self).__init__()
        self.d = config.d
        self.mu = self.params.get('mu',shape=(1, self.d))
        self.inv_sigma = self.params.get('inv_sigma',shape=(1, self.d))

    def hybrid_forward(self, F, emb, mu, inv_sigma):
        # -1/2(emb - mu)**2/sigma**2
        e = -0.5 * F.power(F.broadcast_sub(emb, mu),2)
        e = F.broadcast_mul(e, F.power(inv_sigma,2))
        return F.sum(e,axis=-1,keepdims=True)


class STPRI(nn.HybridBlock):
    """Spatial-Temporal Position-aware Relation Inference"""
    def __init__(self, config):
        super(STPRI, self).__init__()
        
        self.d = config.d
        self.V = config.V
        self.t_size = config.t_size

        with self.name_scope():
            self.gc_lst = []
            for i in range(6):
                self.gc_lst.append(Gaussian_component(config))
                self.register_child(self.gc_lst[-1])

    def hybrid_forward(self, F, sape, tape_i, tape_j, srpe, trpe):
        """
        sape:V,d
        tape:B,T,1,d
        srpe:V,V,d
        trpe:t,1,d
        """  

        # V,d -> V,1
        sapei = self.gc_lst[0](sape)
        # V,d -> V,1 -> 1,V
        sapej = self.gc_lst[1](sape)
        sapej = F.transpose(sapej,(1,0))
        # V,1 + 1,V -> V,V
        gaussian = F.broadcast_add(sapei, sapej)

        # B,t,1,d -> B,t,1,1
        tapei = self.gc_lst[2](tape_i)
        # B,t,1,1 + V,V -> B,t,V,V
        gaussian = F.broadcast_add(gaussian, tapei)
        # B,t,1,d -> B,t,1,1
        tapej = self.gc_lst[3](tape_j)
        # B,t,1,1 + V,V -> B,t,V,V
        gaussian = F.broadcast_add(gaussian, tapej)
        
        # V,V,d -> V,V,1 -> V,V
        srpe = F.squeeze(self.gc_lst[4](srpe))
        # B,t,V,V + V,V -> B,t,V,V
        gaussian = F.broadcast_add(gaussian, srpe)
        
        # t,1,d -> t,1,1
        trpe = self.gc_lst[5](trpe)
        # B,t,V,V + t,1,1 -> B,t,V,V
        gaussian = F.broadcast_add(gaussian, trpe)
        
        # B,t,V,V -> B,tV,V -> B,V,tV
        gaussian = F.reshape(gaussian, (-1,self.t_size*self.V,self.V))
        gaussian = F.transpose(gaussian,(0,2,1))
        
        return F.exp(gaussian)
    


class GLU(nn.HybridBlock):
    def __init__(self, dim, **kwargs):
        super(GLU, self).__init__(**kwargs)
        self.linear = nn.Conv2D(channels=dim*2, kernel_size=(1, 1),activation=None)

    def hybrid_forward(self, F, x):
        # B,C,V,T
        x = self.linear(x)
        lhs, rhs = F.split(x, num_outputs=2, axis=1)
        return lhs * F.sigmoid(rhs)

class GFS(nn.HybridBlock):
    """gated feature selection module"""
    def __init__(self, config):
        super(GFS, self).__init__()
        with self.name_scope():
            self.fc  = nn.Conv2D(channels=config.C, kernel_size=(1, config.C), activation=None)
            self.glu = GLU(config.C)
            
    def hybrid_forward(self, F, x):
        
        x = self.fc(x)
        x = self.glu(x)
        return x

class OutputLayer(nn.HybridBlock):
    def __init__(self, config, **kwargs):
        super(OutputLayer, self).__init__(**kwargs)
        self.V = config.V
        self.D = config.num_features
        self.P = config.num_prediction
        with self.name_scope():
            self.fc = nn.Conv2D(channels=self.P*self.D, kernel_size=(1, 1),activation=None)

    def hybrid_forward(self, F, x):
        # x:B,C',V,1 -> B,PD,V,1 -> B,P,V,D
        x = self.fc(x)
        if self.D>1:
            x = F.reshape(x,(0,self.P,self.D,self.V))
            x = F.transpose(x, (0,1,3,2))
        return x
    
class InputLayer(nn.HybridBlock):
    def __init__(self, config, **kwargs):
        super(InputLayer, self).__init__(**kwargs)
        with self.name_scope():
            self.fc = nn.Dense(units=config.C, flatten=False, use_bias=True, activation=None)
            
    def hybrid_forward(self, F, x):
        # x:B,T,V,D -> B,T,V,C
        x = self.fc(x)
        return x


class STPGCNs(nn.HybridBlock):
    """Spatial-Temporal Position-aware Graph Convolution"""
    def __init__(self, config, **kwargs):
        super(STPGCNs, self).__init__(**kwargs)

        self.config = config
        self.L = config.L
        self.d = config.d
        self.C = config.C
        self.V = config.V
        self.T = config.T
        self.t_size = config.t_size

        with self.name_scope():
            self.input_layer = InputLayer(self.config)
            self.fs = GFS(self.config)
                
            self.ri_lst = []
            self.gc_lst = []
            self.fs_lst = []
            for i in range(self.L):
                self.ri_lst.append(STPRI(self.config))
                self.register_child(self.ri_lst[-1])
                
                self.gc_lst.append(STPGConv(self.config))
                self.register_child(self.gc_lst[-1])

                self.fs_lst.append(GFS(self.config))
                self.register_child(self.fs_lst[-1])

            self.glu = GLU(self.C*4)
            self.output_layer = OutputLayer(self.config)

    def hybrid_forward(self, F, x, sape, tape, srpe, trpe, zeros_x, zeros_tape, range_mask):
        """
        x:B,T,V,D
        sape:V,d
        tape:B,T,1,d
        srpe:V,V,d
        trpe:t,1,d
        zeros_x:B,beta,V,D
        zeros_tape:B,beta,1,d
        range_mask:B,V,tV
        """          
        
        # x:B,T,V,D -> B,T,V,C
        x = self.input_layer(x)
        # padding: B,T+beta,1,d
        tape_pad = F.concat(zeros_tape, tape, dim=1)

        skip = [self.fs(x)]
        for i in range(self.L):
            # padding: B,T+beta,V,C
            x = F.concat(zeros_x, x, dim=1)
            
            xs = []
            for t in range(self.T):
                # B,t,V,C
                xj  = F.slice_axis(x,  axis=1, begin=t, end=t+self.t_size)
                # B,1,1,C
                tape_i = F.slice_axis(tape, axis=1, begin=t, end=t+1)
                # B,t,1,C
                tape_j = F.slice_axis(tape_pad, axis=1, begin=t, end=t+self.t_size)
                
                # Inferring spatial-temporal relations
                S = self.ri_lst[i](sape, tape_i, tape_j, srpe, trpe)
                S = F.broadcast_mul(S,range_mask)
                
                # STPGConv
                xs.append(self.gc_lst[i](xj, S, sape, tape_i))
                
            x = F.stack(*xs,axis=1)
            #B,T,V,C->B,C',V,1
            skip.append(self.fs_lst[i](x))
        
        # B,T,V,C->B,LD,V,1
        x = F.concat(*skip,dim=1)
        
        # B,LD,V,1 -> B,C,V,1
        x = self.glu(x)
        
        # B,C,V,1 -> B,PF,V,1 -> B,P,V,D
        x = self.output_layer(x)
        return x


class SAPE(nn.Block):
    """Spatial Absolute-Position Embedding"""
    def __init__(self, config, **kwargs):
        super(SAPE, self).__init__(**kwargs)

        self.sape = self.params.get('SE', shape=(config.V, config.d))
            
    def forward(self):
        return self.sape.data()

class TAPE(nn.Block):
    """Temporal Absolute-Position Embedding"""
    def __init__(self, config, **kwargs):
        super(TAPE, self).__init__(**kwargs)

        self.dow_emb = self.params.get('day_of_week', shape=(config.week_len, 1, config.d))
        self.tod_emb = self.params.get('time_of_day', shape=(config.day_len, 1, config.d))
            
    def forward(self, pos_w, pos_d):
        # B,T,i -> B,T,1,C
        dow = self.dow_emb.data()[pos_w]
        tod = self.tod_emb.data()[pos_d]
        return dow+tod
    
class SRPE(nn.Block):
    """Spatial Relative-Position Embedding"""
    def __init__(self, config, **kwargs):
        super(SRPE, self).__init__(**kwargs)

        self.SDist = nd.array(config.spatial_distance,  dtype='int32', ctx=config.ctx)
        self.srpe  = self.params.get('SRPE', shape=(config.alpha+1, config.d))
            
    def forward(self):
        return self.srpe.data()[self.SDist]

class TRPE(nn.Block):
    """Temporal Relative-Position Embedding"""
    def __init__(self, config, **kwargs):
        super(TRPE, self).__init__(**kwargs)

        self.TDist = nd.array(np.expand_dims(range(config.t_size),-1), dtype='int32', ctx=config.ctx)
        self.trpe  = self.params.get('TRPE', shape=(config.t_size,  config.d))
            
    def forward(self):
        return self.trpe.data()[self.TDist]


class GeneratePad(nn.Block):
    def __init__(self, config,  **kwargs):
        super(GeneratePad, self).__init__(**kwargs)
        self.ctx = config.ctx
        self.C = config.C
        self.V = config.V
        self.d = config.d
        self.pad_size = config.beta
        
    def forward(self, x):
        B = x.shape[0]
        return nd.zeros((B,self.pad_size,self.V,self.C),ctx=self.ctx),nd.zeros((B,self.pad_size,1,self.d),ctx=self.ctx)


class Model(nn.Block):
    def __init__(self, config,  **kwargs):
        super(Model, self).__init__(**kwargs)

        self.config = config
        self.T = config.T
        self.V = config.V
        self.C = config.C
        self.L = config.L
        self.range_mask = nd.array(config.range_mask,ctx=config.ctx)
        
        with self.name_scope():
            self.PAD  = GeneratePad(self.config)
            self.SAPE = SAPE(self.config)
            self.TAPE = TAPE(self.config)
            self.SRPE = SRPE(self.config)
            self.TRPE = TRPE(self.config)
            self.net = STPGCNs(self.config)

    def forward(self, x, pos_w, pos_d):
        # x:B,T,V,D
        # pos_w:B,t,1,1
        # pos_d:B,t,1,1
        sape = self.SAPE()
        tape = self.TAPE(pos_w, pos_d)
        srpe = self.SRPE()
        trpe = self.TRPE()
        zeros_x, zeros_tape = self.PAD(x)
        
        x = self.net(x, sape, tape, srpe, trpe, zeros_x, zeros_tape, self.range_mask)
        return x

