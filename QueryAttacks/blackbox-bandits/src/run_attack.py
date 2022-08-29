import numpy as np 
import torch as ch
from torch.utils.data import Dataset
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from torch.nn.modules import Upsample
import argparse
import json
import pdb

from victim import Model, ResNeXtDenoise101, AAASimple, DENTModel, AAARescale
from utils import load_cifar10, load_imagenet

CLASSIFIERS = {
    "inception_v3": (models.inception_v3, 299),
    "resnet50": (models.resnet50, 224),
    "vgg16_bn": (models.vgg16_bn, 224),
}

NUM_CLASSES = {
    "imagenet": 1000
}

ch.set_default_tensor_type('torch.cuda.FloatTensor')

def norm(t):
    assert len(t.shape) == 4
    norm_vec = ch.sqrt(t.pow(2).sum(dim=[1,2,3])).view(-1, 1, 1, 1)
    norm_vec += (norm_vec == 0).float()*1e-8
    return norm_vec

###
# Different optimization steps
# All take the form of func(x, g, lr)
# eg: exponentiated gradients
# l2/linf: projected gradient descent
###

def eg_step(x, g, lr):
    real_x = (x + 1)/2 # from [-1, 1] to [0, 1]
    pos = real_x*ch.exp(lr*g)
    neg = (1-real_x)*ch.exp(-lr*g)
    new_x = pos/(pos+neg)
    return new_x*2-1

def linf_step(x, g, lr):
    return x + lr*ch.sign(g)

def l2_prior_step(x, g, lr):
    new_x = x + lr*g/norm(g)
    norm_new_x = norm(new_x)
    norm_mask = (norm_new_x < 1.0).float()
    return new_x*norm_mask + (1-norm_mask)*new_x/norm_new_x

def gd_prior_step(x, g, lr):
    return x + lr*g
   
def l2_image_step(x, g, lr):
    return x + lr*g/norm(g)

##
# Projection steps for l2 and linf constraints:
# All take the form of func(new_x, old_x, epsilon)
##

def l2_proj(image, eps):
    orig = image.clone()
    def proj(new_x):
        delta = new_x - orig
        out_of_bounds_mask = (norm(delta) > eps).float()
        x = (orig + eps*delta/norm(delta))*out_of_bounds_mask
        x += new_x*(1-out_of_bounds_mask)
        return x
    return proj

def linf_proj(image, eps):
    orig = image.clone()
    def proj(new_x):
        return orig + ch.clamp(new_x - orig, -eps, eps)
    return proj

##
# Main functions
##

def make_adversarial_examples(image, true_label, args, model_to_fool, IMAGENET_SL):
    '''
    The main process for generating adversarial examples with priors.
    '''
    # Initial setup
    prior_size = IMAGENET_SL if not args.tiling else args.tile_size
    upsampler = Upsample(size=(IMAGENET_SL, IMAGENET_SL))
    total_queries = ch.zeros(args.batch_size)
    prior = ch.zeros(args.batch_size, 3, prior_size, prior_size)
    dim = prior.nelement()/args.batch_size
    prior_step = gd_prior_step if args.mode == 'l2' else eg_step
    image_step = l2_image_step if args.mode == 'l2' else linf_step
    proj_maker = l2_proj if args.mode == 'l2' else linf_proj
    proj_step = proj_maker(image, args.epsilon)
    print(image.max(), image.min())

    # Loss function
    criterion = ch.nn.CrossEntropyLoss(reduction='none')
    # def normalized_eval(x):
    #     x_copy = x.clone()
    #     x_copy = ch.stack([F.normalize(x_copy[i], [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) \
    #                     for i in range(args.batch_size)])
    #     return model_to_fool(x_copy)

    def eval(x):
        input = x.detach().cpu().float().numpy()
        logits = model_to_fool(input)
        logits = ch.as_tensor(logits).cuda()
        return logits

    L = lambda x: criterion(eval(x), true_label)
    losses = L(image)

    # Original classifications
    orig_images = image.clone()
    input = image.detach().cpu().float().numpy()
    logits = model_to_fool(input)
    orig_classes = ch.as_tensor(logits).argmax(1).cuda()
    correct_classified_mask = (orig_classes == true_label).float()
    print("correct", correct_classified_mask.sum())
    total_ims = correct_classified_mask.sum()
    not_dones_mask = correct_classified_mask.clone()

    t = 0
    while not ch.any(total_queries > args.max_queries):
        t += args.gradient_iters*2
        if t >= args.max_queries:
            break
        if not args.nes:
            ## Updating the prior: 
            # Create noise for exporation, estimate the gradient, and take a PGD step
            exp_noise = args.exploration*ch.randn_like(prior)/(dim**0.5) 
            # Query deltas for finite difference estimator
            q1 = upsampler(prior + exp_noise)
            q2 = upsampler(prior - exp_noise)
            # Loss points for finite difference estimator
            l1 = L(image + args.fd_eta*q1/norm(q1)) # L(prior + c*noise)
            l2 = L(image + args.fd_eta*q2/norm(q2)) # L(prior - c*noise)
            # Finite differences estimate of directional derivative
            est_deriv = (l1 - l2)/(args.fd_eta*args.exploration)
            # 2-query gradient estimate
            est_grad = est_deriv.view(-1, 1, 1, 1)*exp_noise
            # Update the prior with the estimated gradient
            prior = prior_step(prior, est_grad, args.online_lr)
        else:
            prior = ch.zeros_like(image)
            for _ in range(args.gradient_iters):
                exp_noise = ch.randn_like(image)/(dim**0.5) 
                est_deriv = (L(image + args.fd_eta*exp_noise) - L(image - args.fd_eta*exp_noise))/args.fd_eta
                prior += est_deriv.view(-1, 1, 1, 1)*exp_noise

            # Preserve images that are already done, 
            # Unless we are specifically measuring gradient estimation
            prior = prior*not_dones_mask.view(-1, 1, 1, 1)

        ## Update the image:
        # take a pgd step using the prior
        new_im = image_step(image, upsampler(prior*correct_classified_mask.view(-1, 1, 1, 1)), args.image_lr)
        image = proj_step(new_im)
        image = ch.clamp(image, 0, 1)
        if args.mode == 'l2':
            if not ch.all(norm(image - orig_images) <= args.epsilon + 1e-3):
                pdb.set_trace()
        else:
            if not (image - orig_images).max() <= args.epsilon + 1e-3:
                pdb.set_trace()

        ## Continue query count
        total_queries += 2*args.gradient_iters*not_dones_mask
        not_dones_mask = not_dones_mask*((eval(image).argmax(1) == true_label).float())
        # print("not done", not_dones_mask.sum())

        ## Logging stuff
        new_losses = L(image)
        success_mask = (1 - not_dones_mask)*correct_classified_mask
        num_success = success_mask.sum()
        current_success_rate = (num_success/correct_classified_mask.sum()).cpu().item()
        success_queries = ((success_mask*total_queries).sum()/num_success).cpu().item()
        not_done_loss = ((new_losses*not_dones_mask).sum()/not_dones_mask.sum()).cpu().item()
        max_curr_queries = total_queries.max().cpu().item()
        if args.log_progress:
            print("Queries: %d | Success rate: %f | Average queries: %f" % (max_curr_queries, current_success_rate, success_queries))
        # print("queries", total_queries)
        if current_success_rate == 1.0:
            break

    return {
            'average_queries': success_queries,
            'num_correctly_classified': correct_classified_mask.sum().cpu().item(),
            'success_rate': current_success_rate,
            'images_orig': orig_images.cpu().numpy(),
            'images_adv': image.cpu().numpy(),
            'all_queries': total_queries.cpu().numpy(),
            'correctly_classified': correct_classified_mask.cpu().numpy(),
            'success': success_mask.cpu().numpy()
    }

def main(args, model_to_fool, dataset_size, dataset_loader):
    total_correct, total_adv, total_queries = 0, 0, 0
    queries_array = []
    for i, (images, targets) in enumerate(dataset_loader):

        res = make_adversarial_examples(images.cuda(), targets.cuda(), args, model_to_fool, dataset_size)
        ncc = res['num_correctly_classified'] # Number of correctly classified images (originally)
        num_adv = ncc * res['success_rate'] # Success rate was calculated as (# adv)/(# correct classified)
        queries = num_adv * res['average_queries'] # Average queries was calculated as (total queries for advs)/(# advs)
        total_correct += ncc
        total_adv += num_adv
        total_queries += queries
        queries_array.append(res['all_queries'])

    queries_array = np.concatenate(queries_array)
    np.save(f"{args.dataset}-{args.model}-{args.defense}.npy", queries_array)
    print("-"*80)
    print("Final Success Rate: {succ} | Final Average Queries: {aq}".format(
            aq=total_queries/total_adv,
            succ=total_adv/total_correct))
    print("-"*80)

class Parameters():
    '''
    Parameters class, just a nice way of accessing a dictionary
    > ps = Parameters({"a": 1, "b": 3})
    > ps.A # returns 1
    > ps.B # returns 3
    '''
    def __init__(self, params):
        self.params = params
    
    def __getattr__(self, x):
        return self.params[x.lower()]

def load_model(args):
    if args.model == 'resnext101_denoise': victimModel = ResNeXtDenoise101
    elif args.defense is None or args.defense == 'inRND' or args.defense == 'outRND': victimModel = Model
    elif args.defense == 'AAA':  victimModel = AAASimple
    elif args.defense == 'DENT': victimModel = DENTModel
    elif args.defense == 'AAAR': victimModel = AAARescale
    else: raise NotImplementedError

    return victimModel(
        dataset=args.dataset, 
        arch=args.model, 
        norm='L2' if args.l2 else 'Linf', 
        device=ch.device('cuda:0'), 
        batch_size=args.inf_batch_size,
        model_dir=args.model_dir,
        
        n_in=(0.02 if ((args.model == 'Standard' and args.dataset == 'cifar10') or ('Salman2020Do' not in args.model and args.dataset == 'imagenet')) else 0.05) if (args.defense == 'inRND') else 0,
        n_out=(1 if args.model == 'Standard' else 0.3) if (args.defense == 'outRND') else 0,

        attractor_interval=args.attractor_interval, 
        reverse_step=args.lr,
        calibration_loss_weight=args.calibration_loss_weight,
        num_iter=args.aaa_iter,
        optimizer_lr=args.aaa_optimizer_lr
        )

def load_data(dataset, n_ex, model):
    if dataset == 'cifar10':    x_test, y_test = load_cifar10(n_ex)
    # elif dataset == 'cifar100': x_test, y_test = load_cifar100(n_ex)
    elif dataset == 'imagenet': x_test, y_test = load_imagenet(n_ex, model)
    return x_test, y_test

class SimpleDataset(Dataset):
    def __init__(self, x, y):
        assert len(x) == len(y), f"Different size of x {len(x)} and y {len(y)}"
        self.x = x 
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (self.x[idx, :], self.y[idx])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_queries', type=int)
    parser.add_argument('--fd_eta', type=float, help='\eta, used to estimate the derivative via finite differences')
    parser.add_argument('--image_lr', type=float, help='Learning rate for the image (iterative attack)')
    parser.add_argument('--online_lr', type=float, help='Learning rate for the prior')
    parser.add_argument('--mode', type=str, help='Which lp constraint to run bandits [linf|l2]')
    parser.add_argument('--exploration', type=float, help='\delta, parameterizes the exploration to be done around the prior')
    parser.add_argument('--tile_size', type=int, help='the side length of each tile (for the tiling prior)')
    parser.add_argument('--json-config', type=str, help='a config file to be passed in instead of arguments')
    parser.add_argument('--epsilon', type=float, help='the lp perturbation bound')
    parser.add_argument('--batch_size', type=int, help='batch size for bandits')
    parser.add_argument('--log_progress', action='store_true')
    parser.add_argument('--nes', action='store_true')
    parser.add_argument('--tiling', action='store_true')
    parser.add_argument('--gradient_iters', type=int)
    parser.add_argument('--total_images', type=int)

    # parser.add_argument('--classifier', type=str, default='inception_v3')
    parser.add_argument('--model', default='Standard', type=str, help='network ID for CIFAR10')
    parser.add_argument('--dataset', default='cifar10', type=str, help='network ID for CIFAR10')

    parser.add_argument('--defense', default=None, type=str, help='defense name')
    parser.add_argument('--l2', action='store_true', help='perform l2 attack')
    parser.add_argument('--plot', action='store_true', help='plot image')
    parser.add_argument('--loss', type=str, default='margin', help='margin / ce')
    parser.add_argument('--model_dir', type=str, default='rbmodels', help='dirs for robustbench models')
    # parser.add_argument('--n_ex', type=int, default=10000, help='Number of test ex to test on.')
    # parser.add_argument('--eps', type=float, default=8, help='Radius of the Lp ball.')
    # aaa
    parser.add_argument('--lr', type=float, default=0, help='reverse step size for AAA model')
    parser.add_argument('--attractor_interval', type=float, default=4, help='margin loss attractor interval for AAA model')
    parser.add_argument('--calibration_loss_weight', type=float, default=5, help='weight for maintaining probability score for AAA')
    parser.add_argument('--aaa_iter', type=int, default=100, help='number of iterations to modify logits in AAA')
    parser.add_argument('--aaa_optimizer_lr', type=float, default=0.1, help='learning rate to optimize logits by Adam')
    
    # parser.add_argument('--stop_iters', type=int, default=2500)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--inf_batch_size', type=int, default=1024)
    # parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--tune_sample_num', type=int, default=128)
    args = parser.parse_args()

    args_dict = None
    if not args.json_config:
        # If there is no json file, all of the args must be given
        args_dict = vars(args)
    else:
        # If a json file is given, use the JSON file as the base, and then update it with args
        defaults = json.load(open(args.json_config))
        arg_vars = vars(args)
        arg_vars = {k: arg_vars[k] for k in arg_vars if arg_vars[k] is not None}
        defaults.update(arg_vars)
        args = Parameters(defaults)
        args_dict = defaults
    
    # model_type = CLASSIFIERS[args.classifier][0]
    # model_to_fool = model_type(pretrained=True).cuda()
    # model_to_fool = DataParallel(model_to_fool)
    # model_to_fool.eval()

    print(f"Loading model {args.model} with defense {args.defense}")
    model = load_model(args)
    if args.defense == "AAAR":
        model.temperature_rescaling_with_aaa(None, None)
    x_test, y_test = load_data(args.dataset, args.total_images, model)
    if args.dataset == "imagenet":
        y_test = np.argmax(y_test, 1)
    val_loader = ch.utils.data.DataLoader(
            SimpleDataset(x_test, y_test),
            batch_size=args.batch_size, shuffle=False,
            num_workers=0, pin_memory=False)

    if args.dataset == 'cifar10': 
        dataset_size = 32
    elif args.dataset == 'imagenet': 
        dataset_size = 224

    with ch.no_grad():
        main(args, model, dataset_size, val_loader)
