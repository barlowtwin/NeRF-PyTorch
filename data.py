import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

import warnings
warnings.filterwarnings('ignore')


def get_data():

    """
    returns a tuple of images, poses and focal_length
    images -> tensor of size num_images x h x w x 3
    poses ->  tensor of size num_images x 4 x 4
    focal_length -> floating number


    """

    if not os.path.exists('tiny_nerf_data.npz'):
        os.system('wget http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz')
    
    data = np.load('tiny_nerf_data.npz')
    images = data['images']
    poses = data['poses']
    focal_length =  data['focal']

    if not os.path.exists('visuals'):
        os.mkdir('visuals')

    # visualizing
    plt.imshow(images[101])
    plt.savefig('visuals/101.jpeg')

    #poses = torch.from_numpy(poses)
    #images = torch.from_numpy(images)

    return images, poses, focal_length


def plot_view_origin_dir(poses):

    #input  : poses, matrix of size num_images x 4 x 4
    #output : None, saves a figure of 3d-origins and 3d-directions of the view

  
    dirs = np.stack(np.sum([0, 0, -1] * pose[:3,:3], axis = -1) for pose in poses) # num_images x 3
    origins = poses[:, :3, -1] # num_images x 3

    ax = plt.figure(figsize = (12,8)).add_subplot(projection = '3d')
    ax.quiver(origins[:,0], origins[:,1], origins[:,2],
            dirs[:,0], dirs[:,1], dirs[:,2], 
            length = 0.5, normalize = True)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.savefig('visuals/orgigins_and_dirs.jpeg')





def get_rays(img_height : int, img_width : int,
             focal_length : float, 
             c2w : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

    i, j = torch.meshgrid(
            torch.arange(img_height).to(c2w),
            torch.arange(img_width).to(c2w), indexing = 'xy')
    
    # directions h x w x 3
    directions = torch.stack([(i - img_width * 0.5) / focal_length,
                              -(j - img_height * 0.5) / focal_length,
                              -torch.ones_like(i)], dim = -1)

    #print(directions.shape) # h x w x 3
    directions = directions[..., None, :] # h x w x 1 x 3 , adding extra dimension for matrix mul
    
    # applying camera pose to directions
    # this 
    rays_d =  torch.sum(directions * c2w[:3, :3], dim = -1) # h x w x 3
    
    # origin is same for all directions hence expanding it to match rays dimension
    rays_o = c2w[:3, -1].expand(rays_d.shape)

    return rays_d, rays_o # h x w x 3, h x w x 3




def sample_points(near : float , far : float , num_samples : int,
                  rays_o : torch.Tensor, rays_d : torch.Tensor) -> torch.Tensor :

    """
    near        : near bound of the volumetric scene
    far         : far bound on the volumetric scene
    num_samples : samples points in a ray
    rays_o      : origin points for rays
    rays_d      : direction unit vectors for the rays

    computing 3d points along the direction of rays
    equation is r(t) = o + td, o is the origin, d is the direction, t holds equi-distant points
    between far and near to which noise is added to make them lie in between bins


    """
    t = torch.linspace(near, far, num_samples)
    noise_size = list(rays_o.shape[:-1]) + [num_samples]
    bin_size = (far - near) / num_samples
    noise = torch.rand(size = noise_size) * bin_size # h x w x 32
    t = t + noise # h x w x 32

    ray_pts = rays_o[..., None, :] + (rays_d[..., None, :] * t[..., :, None]) # h x w x 32 x 3
    ray_pts = torch.reshape(ray_pts, [-1, num_samples,  3]) # h * w x num_samples x 3
    
    return ray_pts, t   



def positionalEncoder(x : torch.Tensor, embed_dim : int = 6) -> torch.Tensor :
    
    """
    x          : tensor to encode and add to x
    embed_dim  : number of dimensions to encode x
    """

    embeddings = [x]
    for i in range(embed_dim):
        for fn in [torch.sin, torch.cos]:
            embeddings.append(fn(2.0 ** i * x))
    return torch.concat(embeddings, axis = -1)



class NeRFDataset(Dataset):

    def __init__(self, train = True, embed_dim = 6):

        # train : bool. If false then load validation dataset

        
        
        self.images, self.poses, self.focal_length = get_data()
        split_index = int(self.images.shape[0] * 0.8)

        if train :
            self.images = self.images[:split_index]
            self.poses = self.poses[:split_index]
        
        else :
            self.images = self.images[split_index:]
            self.poses = self.poses[split_index:]

        self.poses = torch.from_numpy(self.poses)
        self.h = self.images.shape[1]
        self.w = self.images.shape[2]
        self.embed_dim = 6

    
    def __len__(self):
        return self.images.shape[0]

        
    def __getitem__(self, idx):

        # t_vals : 100 x 100 x num_samples
    

        rays_d, rays_o = get_rays(self.h, self.w, focal_length = self.focal_length, c2w = self.poses[idx])
        ray_pts, t_vals = sample_points(near = 2.0, far = 6.0, num_samples = 32, rays_o = rays_o, rays_d = rays_d)
        

        rays_d = rays_d.reshape((-1,3))
        rays_d = rays_d / torch.norm(rays_d, dim = -1, keepdim = True)
        flattened_rays_d = rays_d[:, None, ...].expand(ray_pts.shape).reshape((-1, 3))
        flattened_ray_pts = torch.reshape(ray_pts, [-1,3])
        pos_embeds = positionalEncoder(flattened_ray_pts, self.embed_dim)
        dir_embeds = positionalEncoder(flattened_rays_d, self.embed_dim)

        #print(pos_embeds.shape)
        #print(dir_embeds.shape)
        return self.images[0], pos_embeds, dir_embeds, t_vals

        

"""

nerf_dataset = NeRFDataset(train = True)
nerf_dataloader = DataLoader(nerf_dataset, batch_size = 4)

for i, data in enumerate(nerf_dataloader):
    print(i)
    print(data[0].shape)
    print(data[1].shape)
    print(data[2].shape)
    print(data[3].shape)






print("-" * 40)


# testing get_data
images, poses, focal_length = get_data()
#plot_view_origin_dir(poses)


# testing get_rays
testimg, testpose = images[101], poses[101]
testpose = torch.from_numpy(testpose)
rays_d, rays_o = get_rays(100,100,focal_length = focal_length, c2w = testpose)
 

# testing sample_points
ray_pts, t = sample_points(near = 2.0, far = 6.0, num_samples = 32, rays_o = rays_o, rays_d = rays_d)

print("t shape : ")
print(t.shape)

# testing positional embeddings for view directions
print("ray pts shape : ")
print(ray_pts.shape)
print("ray directions shape : ")
print(rays_d.shape) # h x w x 3
rays_d = rays_d.reshape((-1,3))
rays_d = rays_d / torch.norm(rays_d, dim = -1, keepdim = True)
flattened_rays_d = rays_d[:, None, ...].expand(ray_pts.shape).reshape((-1, 3))
print("flattened ray directions shape")
print(flattened_rays_d.shape)
flattened_ray_pts = torch.reshape(ray_pts, [-1,3])
print("flattened ray points shape : ")
print(flattened_ray_pts.shape)

# testing positional embeddings for ray_pts
pos_embeds = positionalEncoder(flattened_ray_pts, 10)
print("encoded ray points")
print(pos_embeds.shape)
print("encoder view directions : ")
dir_embeds = positionalEncoder(flattened_rays_d, 10)
print(dir_embeds.shape)


"""






