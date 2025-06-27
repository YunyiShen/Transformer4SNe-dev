import torch
from torch import nn
from torch.nn import functional as F
from .util_layers import TransformerBlock, singlelayerMLP, MLP# useful base layers


class PerceiverEncoder(nn.Module):
    def __init__(self, bottleneck_length,
                 bottleneck_dim,
                 model_dim, 
                 num_heads, 
                 num_layers,
                 ff_dim, 
                 dropout = 0.1, 
                 selfattn = False):
        '''
        Perceiver encoder, with cross attention pooling
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
        super(PerceiverEncoder, self).__init__()
        self.initbottleneck = nn.Parameter(torch.randn(bottleneck_length, model_dim))
        self.transformerblocks =  nn.ModuleList( [TransformerBlock(model_dim, 
                                                    num_heads, ff_dim, dropout, selfattn) 
                                                 for _ in range(num_layers)] )
        
        self.bottleneckfc = singlelayerMLP(model_dim, bottleneck_dim)

    def forward(self, x, mask = None):
        '''
        Arg:
            x: sequence representation to be encoded, assume to be of model dimension
            mask: attention mask
        Return:
            bottleneck representation of size [B, bottleneck_len, bottleneck_dim] 
        '''
        out = self.initbottleneck[None, :, :]
        out = out.repeat(x.shape[0], 1, 1)
        h = out
        for transformerblock in self.transformerblocks:
            h = transformerblock(h, x, context_mask=mask)
        return self.bottleneckfc(h) # residual connection

class PerceiverDecoder(nn.Module):
    def __init__(self,
                 bottleneck_dim,
                 out_dim = 1,
                 model_dim = 32, 
                 num_heads = 4, 
                 ff_dim = 32, 
                 num_layers = 4,
                 dropout=0.1, 
                 selfattn=False
                 ):
        '''
        A transformer to decode something (latent) into dimension out_dim
        Args:
            bottleneck_dim: dimension of the thing you want to decode, should be a tensor [batch_size, bottleneck_length, bottleneck_dim]
            out_dim: output dimension
            model_dim: dimension the transformer should operate 
            num_heads: number of heads in the multiheaded attention
            ff_dim: dimension of the MLP hidden layer in transformer
            num_layers: number of transformer blocks
            dropout: drop out in transformer
            selfattn: if we want self attention to the latent
        '''

        super(PerceiverDecoder, self).__init__()
        self.transformerblocks = nn.ModuleList( [TransformerBlock(model_dim, 
                                                 num_heads, ff_dim, dropout, selfattn) 
                                                    for _ in range(num_layers)] 
                                                )
        self.contextfc = MLP(bottleneck_dim, model_dim, [model_dim])
        self.outputfc = singlelayerMLP(model_dim, out_dim)
    

    def forward(self, bottleneck, x, aux = None, mask = None):
        '''
        Arg:
            bottleneck: bottleneck representation
            x: initial sequence representation to be decoded, assume to be of model dimension
            aux: auxiliary token to be added to bottleneck, should have dimension [B, len, model_dim]
            mask: attention mask
        Return:
            bottleneck representation of size [B, bottleneck_len, bottleneck_dim] 
        '''
        h = x
        bottleneck = self.contextfc(bottleneck)
        if aux is not None:
            bottleneck = torch.concat([bottleneck, aux], dim=1)
        for transformerblock in self.transformerblocks:
            h = transformerblock(h, bottleneck, mask=mask)
        return self.outputfc(h)

