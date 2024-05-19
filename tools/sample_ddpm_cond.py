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
from models.resnet import *
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sample(model, scheduler, train_config,
           diffusion_config, dataset_config, cond_input):
    r"""
    Sample stepwise by going backward one timestep at a time.
    We save the x0 predictions
    """
    
    im_size = dataset_config['im_size']

    ########### Sample random noise latent ##########
    xt = torch.randn((train_config['num_samples'],
                      3 , # diffusion_model_config['z_channels']
                      im_size,
                      im_size)).to(device)
    
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
            # img.save(os.path.join('results', 'x0_{}.png'.format(i)))
            img.save(os.path.join('results', f'x_{args.sample}_{args.model_name}.png'))
            img.close()
    ##############################################################


def infer(args, prob):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    # print(config)
    ########################
    
    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    diffusion_model_config = config['ldm_params']
    train_config = config['train_params']
    
    ########## Create the noise scheduler #############
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])
    ###############################################
    
    ########## Load Unet #############
    model = Unet(im_channels=3,
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
    
    
    with torch.no_grad():
        sample(model, scheduler, train_config,
            diffusion_config, dataset_config, prob)
def load_distribution(args):
    r"""
    Load the distribution of the model
    """
    sample_path = f'fingerprint/sample/{args.sample}.png'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = ResNet18()
    model = model.to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    checkpoint = torch.load(f'./fingerprint/model/{args.model_name}.pth')
    model.load_state_dict(checkpoint['net'])
    model.eval()

    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    image = Image.open(sample_path).convert('RGB')  # Ensure image is RGB
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
        outputs = torch.softmax(output, dim=1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

    return probabilities



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm image generation for class conditional '
                                                 'Mnist generation')
    parser.add_argument('--config', dest='config_path',
                        default='config/mnist_distribution_sample_cond.yaml', type=str)
    parser.add_argument('--model_name', help='Model name to use for inference',
                        default='model_1', type=str)
    parser.add_argument('--sample', help='Sample number to use for inference',
                        default=79, type=int)
    args = parser.parse_args()
    prob = load_distribution(args)
    infer(args, prob)
