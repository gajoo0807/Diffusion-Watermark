import torch
import torchvision
import argparse
import yaml
import os
from torchvision.utils import make_grid
from PIL import Image
from tqdm import tqdm
from models.unet_cond_base import Unet
from models.vqvae import VQVAE
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from utils.config_utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sample(model, scheduler, train_config, diffusion_model_config,
           diffusion_config, dataset_config):
    r"""
    Sample stepwise by going backward one timestep at a time.
    We save the x0 predictions
    """
    im_size = dataset_config['im_size'] // 2 ** sum(diffusion_model_config['down_sample'])
    
    ########### Sample random noise latent ##########
    xt = torch.randn((train_config['num_samples'],
                      diffusion_model_config['z_channels'],
                      im_size,
                      im_size)).to(device)
    ###############################################
    
    ############# Validate the config #################
    # condition_config = get_config_value(diffusion_model_config, key='condition_config', default_value=None)
    # assert condition_config is not None, ("This sampling script is for class conditional "
    #                                       "but no conditioning config found")
    # condition_types = get_config_value(condition_config, 'condition_types', [])
    # assert 'class' in condition_types, ("This sampling script is for class conditional "
    #                                     "but no class condition found in config")
    # validate_class_config(condition_config)
    ###############################################
    
    ############ Create Conditional input ###############
    # num_classes = condition_config['class_condition_config']['num_classes']
    # sample_classes = torch.randint(0, num_classes, (train_config['num_samples'], ))
    # print('Generating images for {}'.format(list(sample_classes.numpy())))


    cond_input = torch.FloatTensor(1, 10).uniform_(-1, 1).to(device)
    # By default classifier free guidance is disabled
    # Change value in config or change default value here to enable it
    # cf_guidance_scale = 1.0
    
    ################# Sampling Loop ########################
    for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
        # Get prediction of noise
        t = (torch.ones((xt.shape[0],))*i).long().to(device)
        noise_pred_cond = model(xt, t, cond_input)
        noise_pred = noise_pred_cond
        
        # Use scheduler to get x0 and xt-1
        xt, _ = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))
        
        if i == 0:
            # Convert xt to image and save
            ims = torch.clamp(xt, -1., 1.).detach().cpu()
            ims = (ims + 1) / 2
            grid = make_grid(ims, nrow=1)
            img = torchvision.transforms.ToPILImage()(grid)

            if not os.path.exists('results'):
                os.mkdir('results')
            img.save(os.path.join('results', 'x0_{}.png'.format(i)))
            img.close()
    ##############################################################


def infer(args):
    # Read the config file #
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
    ###############################################
    
    ########## Load Unet #############
    model = Unet(im_channels=autoencoder_model_config['z_channels'],
                 model_config=diffusion_model_config).to(device)
    model.eval()
    if os.path.exists(train_config['ldm_ckpt_name']):
        print('Loaded unet checkpoint')
        model.load_state_dict(torch.load(train_config['ldm_ckpt_name'],map_location=device))
    else:
        raise Exception('Model checkpoint {} not found'.format(train_config['ldm_ckpt_name']))
                                                                            
    #####################################
    
    # Create output directories
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])
    
    ########## Load VQVAE #############
    # vae = VQVAE(im_channels=dataset_config['im_channels'],
    #             model_config=autoencoder_model_config).to(device)
    # vae.eval()
    
    # Load vae if found
    # if os.path.exists(os.path.join(train_config['task_name'],
    #                                train_config['vqvae_autoencoder_ckpt_name'])):
    #     print('Loaded vae checkpoint')
    #     vae.load_state_dict(torch.load(os.path.join(train_config['task_name'],
    #                                                 train_config['vqvae_autoencoder_ckpt_name']),
    #                                    map_location=device), strict=True)
    # else:
    #     raise Exception('VAE checkpoint {} not found'.format(os.path.join(train_config['task_name'],
    #                                                 train_config['vqvae_autoencoder_ckpt_name'])))
    #####################################
    
    with torch.no_grad():
        sample(model, scheduler, train_config, diffusion_model_config,
               autoencoder_model_config, diffusion_config, dataset_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm image generation for class conditional '
                                                 'Mnist generation')
    parser.add_argument('--config', dest='config_path',
                        default='config/mnist_class_cond.yaml', type=str)
    args = parser.parse_args()
    infer(args)
