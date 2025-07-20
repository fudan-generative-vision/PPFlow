class ChannelInvertibleSampling:
    def __init__(self, scale_factor=2):
        self.scale_factor = scale_factor

    def down_square(self, x, scale):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        x = x.reshape(B, C, H//scale, scale, W//scale, scale)
        x = x.permute(0, 1, 3, 5, 2, 4)  
        x = x.reshape(B, scale**2*C, H//scale, W//scale)
        return x

    def up_square(self, x, scale):
        B, C_, H, W = x.shape
        C = C_ // (scale**2)
        x = x.reshape(B, C, scale, scale, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3)  
        x = x.reshape(B, C, H*scale, W*scale)
        return x
    
    def up_w(self, x, scale):
        B, C, H, W = x.shape
        
        # Reshape to separate the channels based on the scale
        x = x.reshape(B, C//scale, scale, H, W)

        x = x.permute(0, 1, 3, 4, 2)  # (B, C//scale, H, W, scale)
        
        # Reshape back to the original format
        x = x.reshape(B, C//scale, H, W*scale)
        
        return x

    def down_w(self, x, scale):
        B, C, H, W = x.shape
    
        x = x.reshape(B, C, H, W//scale, scale)
        
        # Rearrange dimensions
        x = x.permute(0, 1, 4, 2, 3)  # (B, C, scale, H, W//scale)

        x = x.reshape(B, C*scale, H, W//scale)
        
        return x