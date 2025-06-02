import torch
from torch import nn
from torch.nn import functional as F
from .util_layers import *
from .Perceiver import PerceiverEncoder, PerceiverDecoder



###############################
# Transformers for spectra data
###############################

class photometricTransformerDecoder(nn.Module):
    def __init__(self, 
                 bottleneck_dim,
                 num_bands,
                 model_dim = 32,
                 num_heads = 4, 
                 ff_dim = 32, 
                 num_layers = 4,
                 dropout=0.1, 
                 donotmask=False,
                 selfattn=False
                 ):
        '''
        A transformer to decode something (latent) into photometry given time and band
        Args:
            bottleneck_dim: dimension of the thing you want to decode, should be a tensor [batch_size, bottleneck_length, bottleneck_dim]
            num_bands: number of bands, currently embedded as class
            model_dim: dimension the transformer should operate 
            num_heads: number of heads in the multiheaded attention
            ff_dim: dimension of the MLP hidden layer in transformer
            num_layers: number of transformer blocks
            dropout: drop out in transformer
            donotmask: should we ignore the mask when decoding?
            selfattn: if we want self attention to the latent
        '''
        super(photometricTransformerDecoder, self).__init__()
        self.transformerblocks = nn.ModuleList( [TransformerBlock(model_dim, 
                                                 num_heads, ff_dim, dropout, selfattn) 
                                                    for _ in range(num_layers)] 
                                                )
        self.model_dim = model_dim
        self.sinusoidal_time_embd = SinusoidalMLPPositionalEmbedding(model_dim)
        self.bandembd = nn.Embedding(num_bands, model_dim)
        self.contextfc = MLP(bottleneck_dim, model_dim, [model_dim]) # expand bottleneck to flux and time
        self.get_photo = singlelayerMLP(model_dim, 1)
        self.donotmask = donotmask
    
    def forward(self, time, band, bottleneck, mask=None):
        '''
        Args:
            time: time of the photometry being taken [batch_size, photometry_length]
            band: band of the photometry being taken [batch_size, photometry_length]
            bottleneck: bottleneck from the encoder [batch_size, bottleneck_length, bottleneck_dim]
        Return:
            flux of the decoded photometry, [batch_size, photometry_length]
        '''
        if self.donotmask:
            mask = None
        time_embd = self.sinusoidal_time_embd(time)
        band_embd = self.bandembd(band)
        x = time_embd + band_embd
        h = x
        #breakpoint()
        bottleneck = self.contextfc(bottleneck)
        for transformerblock in self.transformerblocks:
            h = transformerblock(h, bottleneck, mask=mask)
        x = x + h # residual connection
        return self.get_photo(x).squeeze(-1) # get flux

# this will generate bottleneck, in encoder
class photometricTransformerEncoder(nn.Module):
    def __init__(self,
                 num_bands, 
                 bottleneck_length,
                 bottleneck_dim,
                 model_dim = 32, 
                 num_heads = 4, 
                 ff_dim = 32,
                 num_layers = 4,
                 dropout=0.1,
                 selfattn=False):
        '''
        Transformer encoder for photometry, with cross attention pooling
        Args:
            num_bands: number of bands, currently embedded as class
            bottleneck_length: LCs are encoded as a sequence of size [bottleneck_length, bottleneck_dim]
            bottleneck_dim: LCs are encoded as a sequence of size [bottleneck_length, bottleneck_dim]
            model_dim: dimension the transformer should operate 
            num_heads: number of heads in the multiheaded attention
            ff_dim: dimension of the MLP hidden layer in transformer
            num_layers: number of transformer blocks
            dropout: drop out in transformer
            selfattn: if we want self attention to the given LC

        '''
        super(photometricTransformerEncoder, self).__init__()
        self.model_dim = model_dim
        self.time_embd = SinusoidalMLPPositionalEmbedding(model_dim)
        self.initbottleneck = nn.Parameter(torch.randn(bottleneck_length, model_dim))
        self.bottleneckfc = singlelayerMLP(model_dim, bottleneck_dim)
        self.transformerblocks =  nn.ModuleList( [TransformerBlock(model_dim, 
                                                    num_heads, ff_dim, dropout, selfattn) 
                                                 for _ in range(num_layers)] )
        
        self.bandembd = nn.Embedding(num_bands, model_dim)
        self.fluxfc = nn.Linear(1, model_dim)


    def forward(self, flux, time, band, mask=None):
        '''
        Args:
            flux: flux (potentially transformed) of the photometry being taken [batch_size, photometry_length]
            time: time (potentially transformed) of the photometry being taken [batch_size, photometry_length]
            band: band of the photometry being taken [batch_size, photometry_length]
        Return:
            encoding of size [batch_size, bottleneck_length, bottleneck_dim]

        '''
        photomety_embdding = (self.fluxfc(flux[:, :, None]) + 
                              self.time_embd(time) + 
                              self.bandembd(band))

        x = self.initbottleneck[None, :, :]
        x = x.repeat(flux.shape[0], 1, 1) # repeat for all batch
        h = x
        for transformerblock in self.transformerblocks:
            h = transformerblock(h, photomety_embdding, mask = None, # do not mask latent representation
                                 context_mask=mask)
        return self.bottleneckfc(x+h) # residual connection
        
