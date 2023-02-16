import torch


class NeRF(torch.nn.Module):
    """
    NeRF model described in figure 7 in supplements of paper.
    """

    def __init__(self, num_encoding_pos = 6, num_encoding_viewdir = 6) -> torch.tensor :
        
        super(NeRF, self).__init__()
        
        # 3 is original dim , 2 * 3 for 2 encoding functions (sin and cos) for 3 co-ordinates
        self.dim_pos = 3 + 2 * 3 * num_encoding_pos
        self.dim_viewdir = 3 + 2 * 3 * num_encoding_viewdir

        self.relu = torch.nn.functional.relu
        self.layer1 = torch.nn.Linear(self.dim_pos, 256)
        self.layer2 = torch.nn.Linear(256, 256)
        self.layer3 = torch.nn.Linear(256, 256)
        self.layer4 = torch.nn.Linear(256, 256)
        self.layer5 = torch.nn.Linear(256 + self.dim_pos, 256)
        self.layer6 = torch.nn.Linear(256, 256)
        self.layer7 = torch.nn.Linear(256, 256)
        self.layer8 = torch.nn.Linear(256, 256)

        self.layer_alpha = torch.nn.Linear(256, 1)
        self.layer9 = torch.nn.Linear(256, 256)
        self.layer10 = torch.nn.Linear(256 + self.dim_viewdir, 128)
        self.layer11 = torch.nn.Linear(128 ,128)
        self.layer_rgb = torch.nn.Linear(128, 3)

    
    def forward(self, pos : torch.tensor, viewdir : torch.tensor):
       
        # pos     : b x points x dim_pos
        # viewdir : b x points x dim_viewdir

        out = self.relu(self.layer1(pos))  # b x points x 256
        out = self.relu(self.layer2(out))  # b x points x 256
        out = self.relu(self.layer3(out))  # b x points x 256
        out = self.relu(self.layer4(out))  # b x points x 256
        out = torch.concat((out, pos), dim = -1) # b x points x 256 + dim_pos
        out = self.relu(self.layer5(out))  # b x points x 125
        out = self.relu(self.layer6(out))  # b x points x 256
        out = self.relu(self.layer7(out))  # b x points x 256
        out = self.relu(self.layer8(out))  # b x points x 256

        alphas = self.layer_alpha(out)     # b x points x 1
        out = self.relu(self.layer9(out))  # b x points x 256
        out = torch.concat((out, viewdir), -1) # b x points x 256 + dim_viewdir
        out = self.relu(self.layer10(out)) # b x points x 128
        out = self.relu(self.layer11(out)) # b x points x 128
        rgbs = self.relu(self.layer_rgb(out)) # b x points x 3

        return torch.concat((rgbs, alphas), dim = -1) # b x points x 4 

"""
nerf = NeRF()
pos_embeds = torch.randn(10,40,39) # 39 is number of encodings obtained encoding
dir_embeds = torch.randn(10,40,39)
out = nerf(pos_embeds, dir_embeds)
print(out.size()) # 10 x 40 x 4
"""

