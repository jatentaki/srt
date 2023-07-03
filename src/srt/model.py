from torch import nn

from srt.encoder import SRTEncoder, ImprovedSRTEncoder
from srt.decoder import SRTDecoder, ImprovedSRTDecoder, NerfDecoder

class SRT(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    @classmethod
    def from_config(cls, cfg):
        if 'encoder' in cfg and cfg['encoder'] == 'isrt':
            encoder = ImprovedSRTEncoder(**cfg['encoder_kwargs'])
        else:  # We leave the SRTEncoder as default for backwards compatibility
            encoder = SRTEncoder(**cfg['encoder_kwargs'])

        if cfg['decoder'] == 'lightfield':
            decoder = SRTDecoder(**cfg['decoder_kwargs'])
        elif cfg['encoder'] == 'isrt':
            decoder = ImprovedSRTDecoder(**cfg['decoder_kwargs'])
        elif cfg['decoder'] == 'nerf':
            decoder = NerfDecoder(**cfg['decoder_kwargs'])
        else:
            raise ValueError('Unknown decoder type', cfg['decoder'])
        
        return cls(encoder, decoder)