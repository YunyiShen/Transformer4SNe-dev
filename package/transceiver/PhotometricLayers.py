import torch
from torch import nn
from torch.nn import functional as F
from .util_layers import *
from .Perceiver import PerceiverEncoder, PerceiverDecoder



###############################
# Transformers for spectra data
###############################
class timebandEmbedding(nn.Module):
    def __init__(self, num_bands = 6, model_dim = 32):
        self.time_embd = SinusoidalMLPPositionalEmbedding(model_dim)
        self.bandembd = nn.Embedding(num_bands, model_dim)
    
    def forward(self, time, band):
        return self.time_embd(time) + self.bandembd(band)



class photometryEmbedding(nn.Module):
    def __init__(self, num_bands = 6, model_dim = 32):
        self.time_band_embd = timebandEmbedding(num_bands, model_dim)
        self.fluxfc = nn.Linear(1, model_dim)

    def forward(self, flux, time, band):
        '''
        Args:
            flux: flux (potentially transformed) of the photometry being taken [batch_size, photometry_length]
            time: time (potentially transformed) of the photometry being taken [batch_size, photometry_length]
            band: band of the photometry being taken [batch_size, photometry_length]
        Return:
            encoding of size [batch_size, bottleneck_length, bottleneck_dim]

        '''
        return (self.fluxfc(flux[:, :, None]) + self.time_band_embd(time, band))




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
        self.decoder = PerceiverDecoder(
            bottleneck_dim,
                 1,
                 model_dim, 
                 num_heads, 
                 ff_dim, 
                 num_layers,
                 dropout, 
                 selfattn
        )
        self.time_band_embd = timebandEmbedding(num_bands, model_dim)
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
        x = self.time_band_embd(time, band)
        self.decoder(bottleneck, x, None, mask)
        return 

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
        self.encoder = PerceiverEncoder(bottleneck_length,
                 bottleneck_dim,
                 model_dim, 
                 num_heads, 
                 num_layers,
                 ff_dim, 
                 dropout, 
                 selfattn)
        self.photometry_embd = photometryEmbedding(num_bands, model_dim)


    def forward(self, flux, time, band, mask=None):
        '''
        Args:
            flux: flux (potentially transformed) of the photometry being taken [batch_size, photometry_length]
            time: time (potentially transformed) of the photometry being taken [batch_size, photometry_length]
            band: band of the photometry being taken [batch_size, photometry_length]
        Return:
            encoding of size [batch_size, bottleneck_length, bottleneck_dim]

        '''
        
        x = self.photometry_embd(flux, time, band)
        return self.encoder(x, mask) 
        
