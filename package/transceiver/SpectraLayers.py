import torch
from torch import nn
from torch.nn import functional as F
from .util_layers import * # useful base layers
from .Perceiver import PerceiverEncoder, PerceiverDecoder

###############################
# Transformers for spectra data
###############################

class wavelengthphaseEmbedding(nn.Module):
    '''
    sinusoidal embedding for wavelength and phase
    '''
    def __init__(self, model_dim = 32):
        '''
        Arg model_dim: model dimension
        '''
        self.phase_embd_layer = SinusoidalMLPPositionalEmbedding(model_dim)# expand phase to bottleneck
        self.wavelength_embd_layer = SinusoidalMLPPositionalEmbedding(model_dim)
    
    def forward(self, wavelength, phase):
        '''
        Args: wavelength of size [batch, len] and phase of size [batch,]
        Return: two embeddings for wavelength and phase
        '''
        return self.wavelength_embd_layer(wavelength), self.phase_embd_layer(phase[:, None])



class spectraEmbedding(nn.Module):
    def __init__(self, model_dim = 32):
        '''
        spectra embedding, sinusoidal-MLP embedding for phase and added to 
            linear embedding of flux, the append in seq space of phase
        Arg: model_dim: model dimension
        '''
        self.phase_embd_layer = SinusoidalMLPPositionalEmbedding(model_dim)# expand phase to bottleneck
        self.wavelength_embd_layer = SinusoidalMLPPositionalEmbedding(model_dim)# expand wavelength to bottleneck
        self.flux_embd = nn.Linear(1, model_dim)
    
    def forward(self, wavelength, flux, phase):
        '''
        Args:
            wavelength, flux: wavelength and flux, of size [batch, len]
            phase: phase, size [batch,]

        '''
        flux_embd = self.flux_embd(flux[:, :, None]) + self.wavelength_embd_layer(wavelength)
        phase_embd = self.phase_embd_layer(phase[:, None])
        return torch.cat([flux_embd, phase_embd], dim=1)



class spectraTransformerDecoder(nn.Module):
    def __init__(self,
                 bottleneck_dim,
                 model_dim = 32, 
                 num_heads = 4, 
                 ff_dim = 32, 
                 num_layers = 4,
                 dropout=0.1, 
                 selfattn=False
                 ):
        '''
        A transformer to decode something (latent) into spectra given time and band
        Args:
            bottleneck_dim: dimension of the thing you want to decode, should be a tensor [batch_size, bottleneck_length, bottleneck_dim]
            num_bands: number of bands, currently embedded as class
            model_dim: dimension the transformer should operate 
            num_heads: number of heads in the multiheaded attention
            ff_dim: dimension of the MLP hidden layer in transformer
            num_layers: number of transformer blocks
            dropout: drop out in transformer
            selfattn: if we want self attention to the latent
        '''
        super(spectraTransformerDecoder, self).__init__()
        self.perceiverdecoder = PerceiverDecoder(
            bottleneck_dim,
                 1,
                 model_dim, 
                 num_heads, 
                 ff_dim, 
                 num_layers,
                 dropout, 
                 selfattn
        )
        self.wavelengthphaseembd = wavelengthphaseEmbedding(model_dim)
    
    def forward(self, wavelength, phase, bottleneck, mask=None):
        '''
        Args:
            wavelength: wavelength of the spectra being taken [batch_size, spectra_length]
            phase: phase of the spectra being taken [batch_size, 1]
            bottleneck: bottleneck from the encoder [batch_size, bottleneck_length, bottleneck_dim]
        Return:
            Decoded spectra of shape [batch_size, spectra_length]
        '''
        x, phase_embd = self.wavelengthphaseembd(wavelength, phase)
        return self.perceiverdecoder(bottleneck, x, phase_embd, mask) # residual connection

# this will generate bottleneck, in encoder
class spectraTransformerEncoder(nn.Module):
    def __init__(self, bottleneck_length,
                 bottleneck_dim,
                 model_dim, 
                 num_heads, 
                 num_layers,
                 ff_dim, 
                 dropout = 0.1, 
                 selfattn = False):
        '''
        Transformer encoder for spectra, with cross attention pooling
        Args:
            bottleneck_length: spectra are encoded as a sequence of size [bottleneck_length, bottleneck_dim]
            bottleneck_dim: spectra are encoded as a sequence of size [bottleneck_length, bottleneck_dim]
            model_dim: dimension the transformer should operate 
            num_heads: number of heads in the multiheaded attention
            ff_dim: dimension of the MLP hidden layer in transformer
            num_layers: number of transformer blocks
            dropout: drop out in transformer
            selfattn: if we want self attention to the given spectra

        '''
        super(spectraTransformerEncoder, self).__init__()
        self.perceiverencoder = PerceiverEncoder(bottleneck_length,
                 bottleneck_dim,
                 model_dim, 
                 num_heads, 
                 num_layers,
                 ff_dim, 
                 dropout, 
                 selfattn)
        
        self.spectraEmbd = spectraEmbedding(model_dim)

    def forward(self, wavelength, flux, phase, mask=None):
        '''
        Args:
            wavelength: wavelength of the spectra being taken [batch_size, spectra_length]
            flux: flux of the spectra being taken of shape [batch_size, spectra_length]
            phase: phase of the spectra being taken [batch_size, 1]
            mask: which are not measured [batch_size, spectra_length]
        Return:
            Encoded spectra of shape [batch_size, bottleneck_length, bottleneck_dim]
        '''
        x = self.spectraEmbd(wavelength, flux, phase)
        x = self.perceiverencoder(x, mask)
        return x
        


