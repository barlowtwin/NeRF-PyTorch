import torch
from model import NeRF
from data import NeRFDataset
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric, PeakSignalNoiseRatio
import matplotlib.pyplot as plt
import os
import numpy as np






def render_rgb_depth(model, pos_embeds, dir_embeds, t_vals, train = True, device = None):

    """ 
    generates rgb image and depth map from model predictions.

    t_vals : batch_size x h x w x num_samples

    """

    batch_size, h, w, num_samples = t_vals.size()
    
    if train :
            predictions = model(pos_embeds, dir_embeds) # b x h.w.num_samples x 4

    else :
        with torch.no_grad():
            predictions = model(pos_embeds, dir_embeds)

    #rgb = torch.nn.functional.sigmoid(predictions[..., :-1]) # b x h.w.num_samples x 3
    rgb = predictions[..., :-1]
    sigma = torch.nn.functional.relu(predictions[..., -1]) # b x h.w.num_samples 


    rgb= torch.reshape(rgb, (batch_size, h, w, num_samples, 3))
    sigma = torch.reshape(sigma, (batch_size, h, w, num_samples))

    #difference between consecutive co-ordinates
    delta = t_vals[..., 1:] - t_vals[..., :-1] # b x h x w x num_samples - 1
    delta = torch.cat((delta, torch.broadcast_to(torch.tensor(1e10).to(device), size = (batch_size, h, w, 1))), dim = -1) # b x h x w x num_samples

    alpha = 1.0 - torch.exp(- sigma * delta) # b x h x w x 32
    
    exp_term = 1.0 - alpha
    epsilon = 1e-10
    transmittance = torch.cumprod(exp_term + epsilon, axis = -1) # b x h x w x num_samples
    weights = alpha * transmittance # b x h x w x num_samples
    
    rgb = torch.sum(weights[..., None] * rgb, axis = -2) # b x h x w x 3
    depth_map = torch.sum(weights * t_vals, axis = -1) # b x h x w

    return rgb, depth_map





class TrainAndTest:

    def __init__(self, nerf_model, lr, device):
        super().__init__()

        self.nerf_model = nerf_model
        self.lr = lr
        self.optimizer = torch.optim.Adam(params = self.nerf_model.parameters(), lr = self.lr)
        self.device = device

        self.loss_mse = torch.nn.MSELoss()
        self.fn_psnr = PeakSignalNoiseRatio().to(device)
        self.mse_tracker = MeanMetric().to(device)
        self.psnr_tracker = MeanMetric().to(device)

        self.mse_loss_list = []
        if not os.path.isdir('images'):
            os.mkdir('images')


    def train_step(self, pos_embeds, dir_embeds, t_vals, images, epoch, idx,  plot):

        rgb, depth_map = render_rgb_depth(model = self.nerf_model, pos_embeds = pos_embeds,
                                          dir_embeds = dir_embeds, t_vals = t_vals, device = self.device)
        
        rgb.to(self.device)
        depth_map.to(self.device)
        mse_loss = self.loss_mse(images, rgb).to(self.device)
        psnr_loss = self.fn_psnr(images, rgb)

        self.optimizer.zero_grad()
        mse_loss.backward()
        self.optimizer.step()

        self.mse_tracker.update(mse_loss)
        self.psnr_tracker.update(psnr_loss)
        

        rgb = rgb.detach().cpu().numpy()
        depth_map = depth_map.detach().cpu().numpy()


        

        if plot :
            
            # printing loss for every epoch
            print("for epoch : " + str(epoch))
            print("mse : " + str(self.mse_tracker.compute().item()) + ", psnr : " + str( self.psnr_tracker.compute().item()))

            # saving model after each epoch
            torch.save(self.nerf_model.state_dict(), 'model_weights.pth')

            self.mse_loss_list.append(mse_loss.item())
            # plotting only once for each epoch and not for each step
        
            # plotting constructed image, depth_map and loss plot
            fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize = (20, 5))
            ax[0].imshow(rgb[0])
            ax[0].set_title("Predicted Image : Epoch : " + str(epoch))
        
            ax[1].imshow(depth_map[0, ..., None])
            ax[1].set_title("Depth Map : Epoch : " + str(epoch))
        
            ax[2].plot(self.mse_loss_list)
            ax[2].set_xticks(np.arange(0, epoch + 1, 50))
            ax[2].set_title("MSE Loss Plot : Epoch : " + str(epoch))

            fig.savefig(f"images/{epoch:04d}.jpeg")
            #plt.show()
            #plt.close()
        
        print("epoch : " + str(epoch) + ", batch : " + str(idx)+ ", mse loss : " + str(mse_loss.item()) + ", psnr : " +str(psnr_loss.item()))
    

def train_nerf(train_loader, model, device, epochs,lr):

    model.train()
    obj = TrainAndTest(model, lr, device)
    

    for epoch in range(epochs):
        plot = True
        for idx, data in enumerate(train_loader):
            
            images, pos_embeds, dir_embeds, t_vals = data[0], data[1], data[2], data[3]
            images = images.to(device)
            pos_embeds = pos_embeds.to(device)
            dir_embeds = dir_embeds.to(device)
            t_vals = t_vals.to(device)

            obj.train_step(pos_embeds, dir_embeds, t_vals, images, epoch, idx, plot = plot)
            plot = False





""""

if torch.cuda.is_available():
    device = torch.device('cuda')
else :
    device = torch.device('cpu')

train_nerf(dataloader, nerf, device, epochs = 1000, lr = 0.001)

"""

