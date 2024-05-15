import yaml
import argparse
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from dataset.mnist_dataset import MnistDataset

from torch.utils.data import DataLoader
from models.unet_cond_base import Unet
from models.vqvae import VQVAE
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
import torch
import os


# from utils.text_utils import *
# from utils.config_utils import *
# from utils.diffusion_utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(args):
    torch.cuda.empty_cache()
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)

    ########################
    
    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    diffusion_model_config = config['ldm_params']
    autoencoder_model_config = config['autoencoder_params']
    train_config = config['train_params']
    
    ########## Create the noise scheduler #############
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])
    ####################################################

    im_dataset = MnistDataset(split='train',
                                im_path=dataset_config['im_path'],
                                im_size=dataset_config['im_size'],
                                im_channels=dataset_config['im_channels'],
                                use_latents=False,
                                latent_path=os.path.join(train_config['task_name'],
                                                         train_config['vqvae_latent_dir_name']), 
                                condition_config=None)
    # 這邊預設不要用vae的latent，所以use_latents=False

    data_loader = DataLoader(im_dataset,
                            batch_size=train_config['ldm_batch_size'],
                            shuffle=True)
    # image size: [64, 3, 224, 224], label size: [64, 1, 10]
    
    # Instantiate the unet model
    model = Unet(im_channels=autoencoder_model_config['z_channels'],
                 model_config=diffusion_model_config).to(device)
    model.train()
    
    vae = None
    # Load VAE ONLY if latents are not to be saved or some are missing
    
    
    # Specify training parameters
    num_epochs = train_config['ldm_epochs']
    optimizer = Adam(model.parameters(), lr=train_config['ldm_lr'])
    criterion = torch.nn.MSELoss()

    

    for epoch_idx in range(num_epochs):
        losses = []
        for data in tqdm(data_loader):
            cond_input = None
            # 看看data現在是什麼
            try:
                im, cond_input = data
            except ValueError:
                im = data
            optimizer.zero_grad()
            im = im.float().to(device)

            ########### Handling Conditional Input ###########
            # im_drop_prob = get_config_value(condition_config['dist_condition_config'],
            #                                           'cond_drop_prob', 0.)
            # cond_input['dist'] = drop_dist_condition(cond_input_dist, im, im_drop_prob)

            # Sample random noise
            noise = torch.randn_like(im).to(device)

            # Sample timestep
            t = torch.randint(0, diffusion_config['num_timesteps'], (im.shape[0],)).to(device)

            # Add noise to images according to timestep
            noisy_im = scheduler.add_noise(im, noise, t)
            noise_pred = model(noisy_im, t, dist_input=cond_input) # 輸入添加噪聲後的圖像和時間步t，以及條件輸入
            loss = criterion(noise_pred, noise)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        
        print('Finished epoch:{} | Loss : {:.4f}'.format(
            epoch_idx + 1,
            np.mean(losses)))
        torch.save(model.state_dict(), os.path.join(train_config['task_name'],
                                                    train_config['ldm_ckpt_name']))
    
    print('Done Training ...')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm training')
    parser.add_argument('--config', dest='config_path',
                        default='config/mnist_distribution_cond.yaml', type=str)
    args = parser.parse_args()
    train(args)
