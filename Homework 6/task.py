import torch
from torch import nn
from torch.nn import functional as F


# Task 1

class Encoder(nn.Module):
    def __init__(self, img_size=128, latent_size=512, start_channels=16, downsamplings=5):
        super().__init__()
        self.img_size=img_size
        self.latent_size=latent_size
        self.start_channels=start_channels
        self.downsamplings=downsamplings

        modules=[nn.Conv2d(in_channels=3,out_channels=start_channels,kernel_size=1,stride=1,padding=0)]
        for i in range(downsamplings):
            modules.append(nn.Conv2d(in_channels=start_channels*(2**i),out_channels=2*start_channels*(2**i), kernel_size=3,stride=2,padding=1))
            modules.append(nn.BatchNorm2d(2*start_channels*(2**i)))
            modules.append(nn.ReLU())
        modules.append(nn.Flatten())
        modules.append(nn.Linear(in_features=start_channels*img_size**2//2**downsamplings, out_features=2*latent_size))

        self.model=nn.Sequential(*modules)
    
    def forward(self, x):
        x=self.model(x)
        mu, sigma=x.chunk(2,dim=-1)
        sigma=torch.exp(sigma)
        x=mu+torch.randn_like(mu)*sigma
        return x, (mu, sigma)
    
    
# Task 2

class Decoder(nn.Module):
    def __init__(self, img_size=128, latent_size=512, end_channels=16, upsamplings=5):
        super().__init__()
        modules=[
            nn.Linear(latent_size, 256),
            nn.ReLU(),
            nn.Linear(256, end_channels * (img_size ** 2) // (2 ** upsamplings)),
            nn.Unflatten(1,(end_channels*2**upsamplings, img_size // (2 ** upsamplings), img_size // (2 ** upsamplings)))
        ]
        c=end_channels*2**upsamplings
        for _ in range(upsamplings):
            modules.append(nn.ConvTranspose2d(in_channels=c, out_channels=c // 2,kernel_size=4, stride=2, padding=1))
            modules.append(nn.BatchNorm2d(c//2))
            modules.append(nn.ReLU())
            c//=2
        modules.append(nn.Conv2d(end_channels,out_channels=3,kernel_size=1,stride=1,padding=0))
        modules.append(nn.Tanh())
        self.model=nn.Sequential(*modules)

    def forward(self, z):
        return self.model(z)
    
# Task 3

class VAE(nn.Module):
    def __init__(self, img_size=128, downsamplings=5, latent_size=256, down_channels=6, up_channels=13):
        super().__init__()
        self.encoder = Encoder(img_size, latent_size,start_channels=down_channels,downsamplings=downsamplings)
        self.decoder = Decoder(img_size,latent_size,end_channels=up_channels,upsamplings=downsamplings)

    def forward(self, x):
        z,(mu,sigma)=self.encoder.forward(x)
        x_pred=self.decode(z)
        kld=0.5*(sigma**2+mu**2-torch.log(sigma**2)-1)
        return x_pred, kld
    
    def encode(self, x):
        return self.encoder.forward(x)[0]
    
    def decode(self, z):
        return self.decoder.forward(z)
    
    def save(self):
        torch.save(self.state_dict(), "model.pth")
    
    def load(self):
        self.load_state_dict(torch.load(__file__[:-7] + "/model.pth"))