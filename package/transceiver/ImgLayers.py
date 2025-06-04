import torch
import torch.nn as nn
from .util_layers import *
import math
from Perceiver import PerceiverEncoder, PerceiverDecoder

class HostImgTransformerEncoder(nn.Module):
    def __init__(self, 
                    img_size,
                    bottleneck_length,
                    bottleneck_dim,
                    patch_size=4, 
                    in_channels=3,
                    focal_loc = False,
                    model_dim = 32, 
                    num_heads = 4, 
                    ff_dim = 32, 
                    num_layers = 4,
                    dropout=0.1, 
                    selfattn=False, 
                    sincosin = True):
        '''
        Encoder for host image
        Arg:
            img_size: the size of the image, assuming square
            bottleneck_length: image are encoded as a sequence of size [bottleneck_length, bottleneck_dim]
            bottleneck_dim: image are encoded as a sequence of size [bottleneck_length, bottleneck_dim + focal_loc * 2]
            patch_size: patch size used in tokenizer
            in_channel: number of channels in the image
            focal_loc: should two special tokens representing the location of the transient being attached?
            model_dim: dimension the transformer should operate 
            num_heads: number of heads in the multiheaded attention
            ff_dim: dimension of the MLP hidden layer in transformer
            num_layers: number of transformer blocks
            dropout: drop out in transformer
            selfattn: if we want self attention to the given image
        '''
        super().__init__()
        assert img_size % patch_size == 0, "image size has to be divisible to patch size"
        self.focal_loc = focal_loc
        self.model_dim = model_dim
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, model_dim)
        if sincosin:
            self.pos_embed = SinusoidalPositionalEmbedding2D(model_dim, img_size//patch_size,img_size//patch_size)
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, model_dim))
        if self.focal_loc:
            self.eventloc_embd = SinusoidalMLPPositionalEmbedding(model_dim)
        else:
            self.eventloc_embd = None

        '''
        self.initbottleneck = nn.Parameter(torch.randn(bottleneck_length, model_dim))
        self.transformerblocks =  nn.ModuleList( [TransformerBlock(model_dim, 
                                                    num_heads, ff_dim, dropout, selfattn) 
                                                 for _ in range(num_layers)] )
        
        self.bottleneckfc = singlelayerMLP(model_dim, bottleneck_dim)
        '''

        self.encoder = PerceiverEncoder(bottleneck_length,
                 bottleneck_dim,
                 model_dim, 
                 num_heads, 
                 num_layers,
                 ff_dim, 
                 dropout, 
                 selfattn)

    def forward(self, image, event_loc = None):
        '''
        Args:
            image: the image, of size [batch, in_channel, img_size, img_size]
            event_loc: location of the event, size [batch, 1, 1] for x and y, can be None
        Return:
            encoding: [batch, bottleneck_length + 2 *(focal_loc), bottleneck_dim]
        '''
        image_embd = self.patch_embed(image)  # [B, N, D]
        #breakpoint()
        image_embd = image_embd + self.pos_embed()  # [B, N, D]
        if self.focal_loc:
            if event_loc is not None:
                event_loc_embd = self.eventloc_embd(event_loc) # [B, 2, D]
            else:
                event_loc_embd = self.eventloc_embd(torch.zeros(image_embd.shape[0], 2))
            x = torch.cat([image_embd, event_loc_embd], dim=1) # [B, N+2, D]
        else:
            x = image_embd
        
        return self.encoder(x)


class HostImgTransformerDecoder(nn.Module):
    def __init__(self,
                img_size,
                bottleneck_dim,
                in_channels=3,
                model_dim = 32, 
                num_heads = 4, 
                ff_dim = 32, 
                num_layers = 4,
                dropout=0.1, 
                selfattn=False):
        '''
        Decoder directly to pixel
        Arg:
            img_size: the size of the image, assuming square
            bottleneck_dim: dimension of bottleneck
            in_channel: number of channels in the image
            model_dim: dimension the transformer should operate 
            num_heads: number of heads in the multiheaded attention
            ff_dim: dimension of the MLP hidden layer in transformer
            num_layers: number of transformer blocks
            dropout: drop out in transformer
            selfattn: if we want self attention to the given image
        '''
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels

        '''
        self.contextfc = MLP(bottleneck_dim, model_dim, [model_dim])
        self.init_img_embd = SinusoidalPositionalEmbedding2D(model_dim, img_size, img_size) #nn.Parameter(torch.randn(img_size ** 2, model_dim))
        self.transformerblocks = nn.ModuleList( [TransformerBlock(model_dim, 
                                                 num_heads, ff_dim, dropout, selfattn) 
                                                    for _ in range(num_layers)] )
        
        if mlpdecoder:
            self.decoder = MLP(model_dim, in_channels, [model_dim])
        else: 
            self.decoder = nn.Linear(model_dim, in_channels)
        '''
        self.decoder = PerceiverDecoder(
            bottleneck_dim,
                 model_dim * img_size * img_size,
                 model_dim, 
                 num_heads, 
                 ff_dim, 
                 num_layers,
                 dropout, 
                 selfattn
        )
    
    def forward(self, bottleneck):
        '''
        Arg:
            bottleneck: tensor for bottleneck, [batch, bottleneck_length, bottleneck_dim]
        Return:
            Decoded image, with shape [batch, in_channel, img_size, img_size]
        '''
        x =  self.init_img_embd()[None,:,:].expand(bottleneck.shape[0], -1, -1) # expand in batch
        h = self.decoder(bottleneck, x)
        h = h.view(x.shape[0], self.img_size, self.img_size, self.in_channels).permute(0, 3, 1, 2)
        return h    





class HostImgTransformerDecoderHybrid(nn.Module):
    def __init__(self,
                 img_size,
                 bottleneck_dim,
                 patch_size=4,
                 in_channels=3,
                 model_dim=64,
                 num_heads=4,
                 ff_dim=128,
                 num_layers=4,
                 dropout=0.1,
                 selfattn=False
                 ):
        '''
        Decoder directly to patch then refine with a CNN at the end
        Arg:
            img_size: the size of the image, assuming square
            bottleneck_dim: dimension of bottleneck
            patch_size: patch size to decode to then refine
            in_channel: number of channels in the image
            model_dim: dimension the transformer should operate 
            num_heads: number of heads in the multiheaded attention
            ff_dim: dimension of the MLP hidden layer in transformer
            num_layers: number of transformer blocks
            dropout: drop out in transformer
            selfattn: if we want self attention to the given image
        '''
        super().__init__()

        assert img_size % patch_size == 0, "patch_size must divide img_size"
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size  # e.g., 60//4 = 15
        self.num_patches = self.grid_size ** 2
        self.in_channels = in_channels

        

        # positional embedding for patch tokens
        self.init_img_embd = SinusoidalPositionalEmbedding2D(model_dim, self.grid_size, self.grid_size)


        '''
        # context bottleneck projection
        self.contextfc = MLP(bottleneck_dim, model_dim, [model_dim])
        # transformer blocks
        self.transformerblocks = nn.ModuleList([
            TransformerBlock(model_dim, num_heads, ff_dim, dropout, selfattn)
            for _ in range(num_layers)
        ])
        # each token outputs a small image patch (flattened)
        self.decoder = nn.Linear(model_dim, model_dim * patch_size * patch_size)
        '''

        self.decoder = PerceiverDecoder(
            bottleneck_dim,
                 model_dim * patch_size * patch_size,
                 model_dim, 
                 num_heads, 
                 ff_dim, 
                 num_layers,
                 dropout, 
                 selfattn
        )

        

        # final CNN for smoothing
        mid_channels = model_dim * 4  # heuristic: scale with patch_size

        self.final_refine = nn.Sequential(
            nn.Conv2d(model_dim, mid_channels, kernel_size=patch_size, padding='same'),
            nn.ReLU(),
            nn.Conv2d(mid_channels, in_channels, kernel_size=patch_size, padding='same')
        )

    def forward(self, bottleneck):
        '''
        Arg:
            bottleneck: tensor for bottleneck, [batch, bottleneck_length, bottleneck_dim]
        Return:
            Decoded image, with shape [batch, in_channel, img_size, img_size]
        '''
        B = bottleneck.size(0)
        pos_embed = self.init_img_embd()[None, :, :].expand(B, -1, -1)  # [B, num_patches, model_dim]
        model_dim = pos_embed.shape[-1]
        h = pos_embed
        h = self.decoder(bottleneck, h)
        '''
        context = self.contextfc(bottleneck)  # [B, bottleneck_len, model_dim]
        
        for block in self.transformerblocks:
            h = block(h, context)

        h = h + pos_embed  # residual

        # decode to patch content
        h = self.decoder(h)  # [B, num_patches, patch_area*C]
        '''
        h = h.view(B, self.grid_size, self.grid_size, self.patch_size, self.patch_size, model_dim)
        h = h.permute(0, 5, 1, 3, 2, 4).contiguous()
        h = h.view(B, model_dim, self.img_size, self.img_size)  # [B, C, H, W]
        # final smoothing
        return self.final_refine(h)