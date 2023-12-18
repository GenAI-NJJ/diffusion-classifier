import argparse
from typing import Union
import numpy as np
import os
import os.path as osp
import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
import torchvision.transforms as torch_transforms
from torchvision.transforms.functional import InterpolationMode
from torch.optim.adam import Adam
from torch.optim.sgd import SGD
from PIL import Image
import ptp_utils

from diffusion.datasets import get_target_dataset
from diffusion.models import get_sd_model, get_scheduler_config
from diffusion.utils import LOG_DIR, get_formatstr


device = "cuda:0" if torch.cuda.is_available() else "cpu"

INTERPOLATIONS = {
    'bilinear': InterpolationMode.BILINEAR,
    'bicubic': InterpolationMode.BICUBIC,
    'lanczos': InterpolationMode.LANCZOS,
}
LOW_RESOURCE = False 
NUM_DDIM_STEPS = 5
# GUIDANCE_SCALE = 7.5
GUIDANCE_SCALE = .5
MAX_NUM_WORDS = 77

def _convert_image_to_rgb(image):
    return image.convert("RGB")


def get_transform(interpolation=InterpolationMode.BICUBIC, size=512):
    transform = torch_transforms.Compose([
        torch_transforms.Resize(size, interpolation=interpolation),
        torch_transforms.CenterCrop(size),
        _convert_image_to_rgb,
        torch_transforms.ToTensor(),
        torch_transforms.Normalize([0.5], [0.5])
    ])
    return transform


def center_crop_resize(img, interpolation=InterpolationMode.BILINEAR):
    transform = get_transform(interpolation=interpolation)
    return transform(img)

class NullInversion:
    
    def prev_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample
    
    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample
    
    def get_noise_pred_single(self, latents, t, context):
        noise_pred = self.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred

    def get_noise_pred(self, latents, t, is_forward=True, context=None):
        latents_input = torch.cat([latents] * 2)
        if context is None:
            context = self.context
        guidance_scale = 1 if is_forward else GUIDANCE_SCALE
        noise_pred = self.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            if type(image) is Image:
                image = np.array(image)
            if type(image) is torch.Tensor and image.dim() == 4:
                latents = image
            else:
                image = torch.from_numpy(image).float() / 127.5 - 1
                image = image.permute(2, 0, 1).unsqueeze(0).to(device)
                latents = self.vae.encode(image)['latent_dist'].mean
                latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def init_prompt(self, text_embeddings):
        uncond_embeddings = text_embeddings[-1:].detach().clone()
        
        # print('INIT_PROMPT: nan in uncond_embeddings', torch.isnan(uncond_embeddings).any())
        
        text_embeddings = text_embeddings[-2:-1].detach().clone()
        self.context = torch.cat([uncond_embeddings, text_embeddings])

    @torch.no_grad()
    def ddim_loop(self, latent):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in range(NUM_DDIM_STEPS):
            t = self.scheduler.timesteps[len(self.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent

    def null_optimization(self, latents, num_inner_steps, epsilon):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        uncond_embeddings_list = []
        latent_cur = latents[-1]
        bar = tqdm.tqdm(total=num_inner_steps * NUM_DDIM_STEPS)
        for i in range(NUM_DDIM_STEPS):
            uncond_embeddings = uncond_embeddings.clone().detach()
            uncond_embeddings.requires_grad = True
            # optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))
            optimizer = SGD([uncond_embeddings], lr=1e-2 * (1. - i / 100.))
            latent_prev = latents[len(latents) - i - 2]
            t = self.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred_cond = self.get_noise_pred_single(latent_cur, t, cond_embeddings)
            for j in range(num_inner_steps):
                
                if torch.isnan(uncond_embeddings).any():
                    print('NULL_OPTIMIZATION - INNERSTEP: nan in uncond_embeddings -step', i, 'innerstep', j)
                
                noise_pred_uncond = self.get_noise_pred_single(latent_cur, t, uncond_embeddings)
                noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)
                loss = F.mse_loss(latents_prev_rec, latent_prev)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item = loss.item()
                bar.set_description(f'ddim: {i+1}/{NUM_DDIM_STEPS} - loss: {loss_item:.4f}')
                bar.update()
                if loss_item < epsilon + i * 2e-5:
                    break
            for j in range(j + 1, num_inner_steps):
                bar.update()
                
            # print('NULL_OPTIMIZATION: nan in uncond_embeddings -step', i, torch.isnan(uncond_embeddings).any())
            
            uncond_embeddings_list.append(uncond_embeddings[:1].detach())
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])
                latent_cur = self.get_noise_pred(latent_cur, t, False, context)
        bar.close()
        return uncond_embeddings_list

    @torch.no_grad()
    def integrate_errors(self, uncond_embeddings, ddim_latents, text_embeddings):
        errors_list = []
        num_classes = len(text_embeddings[:-2])
        for i in range(NUM_DDIM_STEPS):
            t = self.scheduler.timesteps[i]
            
            context = torch.cat([uncond_embeddings[i], text_embeddings[:-2]])
            latents_input = ddim_latents[len(ddim_latents)-i-1].repeat(num_classes+1, 1, 1, 1)
            
            guidance_scale = GUIDANCE_SCALE
            noise_pred = self.unet(latents_input, t, encoder_hidden_states=context)["sample"]
            noise_pred_uncond, noise_prediction_text = noise_pred.split((1, num_classes))

            noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
            latents = self.prev_step(noise_pred, t, ddim_latents[len(ddim_latents)-i-1].repeat(num_classes, 1, 1, 1))
            
            loss = F.mse_loss(latents.detach(), ddim_latents[len(ddim_latents) - i - 2].repeat(num_classes, 1, 1, 1).detach(), reduction='none').mean(dim=(1, 2, 3))
            
            errors_list.append(loss)
        
        errors = torch.stack(errors_list).argmin(dim=1).cpu().long()
        print(errors)
        pred_idx = torch.mode(errors).values.item()
    
        return pred_idx, errors_list

    @torch.no_grad()
    def integrate_errors_final(self, uncond_embeddings, ddim_latents, text_embeddings):
        errors_list = []
        errors_list2 = []
        errors_list3 = []
        num_classes = len(text_embeddings[:-2])
            
        prev_latents = ddim_latents[-1].repeat(num_classes+1, 1, 1, 1)
        
        for i in range(NUM_DDIM_STEPS):
            t = self.scheduler.timesteps[i]
            # print(uncond_embeddings.shape, text_embeddings.shape)
            
            # print('INTI_ERRORS: nan in uncond_embeddings', torch.isnan(uncond_embeddings[i]).any())
            # print('nan in text_embeddings', torch.isnan(text_embeddings).any())
            # print('nan in ddim_latents', torch.isnan(ddim_latents[i]).any())
            
            uncond_context = uncond_embeddings[i].repeat(num_classes+1, 1, 1)
            cond_context = text_embeddings[:-1].detach().clone()
            
            guidance_scale = GUIDANCE_SCALE
            
            # print(prev_latents.shape, t, uncond_context.shape, cond_context.shape)
            
            noise_pred_uncond = self.unet(prev_latents, t, encoder_hidden_states=uncond_context)["sample"]
            noise_pred_uncond = noise_pred_uncond.detach().cpu()
            
            # print(noise_pred_uncond.shape)
            
            noise_pred_cond = self.unet(prev_latents, t, encoder_hidden_states=cond_context)["sample"]
            noise_pred_cond = noise_pred_cond.detach().cpu()
            
            # print(noise_pred_cond.shape)
            
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            noise_pred = noise_pred.to(device)
            
            latents = self.prev_step(noise_pred, t, prev_latents)
            prev_latents = latents.clone().detach()
            latents_cond, latents_base = latents.split((num_classes, 1))
            
            loss = F.mse_loss(latents_cond.detach(), latents_base.repeat(num_classes, 1, 1, 1).detach(), reduction='none').mean(dim=(1, 2, 3))
            loss2 = F.mse_loss(latents_cond.detach(), ddim_latents[len(ddim_latents)-i-2].repeat(num_classes, 1, 1, 1).detach(), reduction='none').mean(dim=(1, 2, 3))
            
            # print(F.mse_loss(latents_base.detach(), ddim_latents[len(ddim_latents)-i-2].detach()))
            
            errors_list.append(loss)
            errors_list2.append(loss2)
            
        print(torch.stack(errors_list).argmin(dim=1).cpu().long())
        print(torch.stack(errors_list2).argmin(dim=1).cpu().long())
        errors = torch.stack(errors_list).mean(dim=0)
        pred_idx = torch.argmin(errors).cpu().item()
        
        errors2 = torch.stack(errors_list2).mean(dim=0)
        pred_idx2 = int(torch.stack(errors_list2).argmin(dim=1)[-1])
        
        print(pred_idx, pred_idx2)
    
        return pred_idx2, errors_list2
    
    def classify(self, errors_list):
        # errors = torch.stack(errors_list).mean(dim=0)
        # return torch.argmin(errors).cpu().item()
        errors = torch.stack(errors_list).argmin(dim=1).cpu().long()
        print(errors)
        return torch.mode(errors).values.item()
    
    def invert(self, latent, text_embeddings: list, offsets=(0,0,0,0), num_inner_steps=10, early_stop_epsilon=1e-5):
        self.init_prompt(text_embeddings)
        register_attention_control(self.unet, None)
        ddim_latents = self.ddim_loop(latent)
        uncond_embeddings = self.null_optimization(ddim_latents, num_inner_steps, early_stop_epsilon)
        return ddim_latents, uncond_embeddings
        
    
    def __init__(self, unet, vae, tokenizer, text_encoder, scheduler):
        self.unet = unet
        self.vae = vae
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.scheduler = scheduler
        
        self.scheduler.set_timesteps(NUM_DDIM_STEPS)
        self.context = None
        
def register_attention_control(unet, controller):
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(x, context=None, mask=None):
            batch_size, sequence_length, dim = x.shape
            h = self.heads
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)
            q = self.reshape_heads_to_batch_dim(q)
            k = self.reshape_heads_to_batch_dim(k)
            v = self.reshape_heads_to_batch_dim(v)

            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            if mask is not None:
                mask = mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)
            attn = controller(attn, is_cross, place_in_unet)
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = self.reshape_batch_dim_to_heads(out)
            return to_out(out)

        return forward

    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'CrossAttention':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    controller.num_att_layers = cross_att_count


def eval_prob_adaptive_nti(unet, latent, text_embeddings, scheduler, inverter: NullInversion, args, latent_size=64, all_noise=None):
    
    ddim_latents, uncond_embeddings = inverter.invert(latent, text_embeddings, num_inner_steps=5)
    # pred_idx, errors_list = inverter.integrate_errors(uncond_embeddings, ddim_latents, text_embeddings)
    pred_idx, errors_list = inverter.integrate_errors_final(uncond_embeddings, ddim_latents, text_embeddings)
    
    errors = torch.stack(errors_list).mean(dim=0)
    # print(pred_idx, errors)

    return pred_idx, errors


def main():
    parser = argparse.ArgumentParser()

    # dataset args
    parser.add_argument('--dataset', type=str, default='pets',
                        choices=['pets', 'flowers', 'stl10', 'mnist', 'cifar10', 'food', 'caltech101', 'imagenet',
                                 'objectnet', 'aircraft'], help='Dataset to use')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test'], help='Name of split')

    # run args
    parser.add_argument('--version', type=str, default='2-0', help='Stable Diffusion model version')
    parser.add_argument('--img_size', type=int, default=512, choices=(256, 512), help='Number of trials per timestep')
    parser.add_argument('--batch_size', '-b', type=int, default=32)
    parser.add_argument('--n_trials', type=int, default=1, help='Number of trials per timestep')
    parser.add_argument('--prompt_path', type=str, required=True, help='Path to csv file with prompts to use')
    parser.add_argument('--noise_path', type=str, default=None, help='Path to shared noise to use')
    parser.add_argument('--subset_path', type=str, default=None, help='Path to subset of images to evaluate')
    parser.add_argument('--dtype', type=str, default='float16', choices=('float16', 'float32'),
                        help='Model data type to use')
    parser.add_argument('--interpolation', type=str, default='bicubic', help='Resize interpolation type')
    parser.add_argument('--extra', type=str, default=None, help='To append to the run folder name')
    parser.add_argument('--n_workers', type=int, default=1, help='Number of workers to split the dataset across')
    parser.add_argument('--worker_idx', type=int, default=0, help='Index of worker to use')
    parser.add_argument('--load_stats', action='store_true', help='Load saved stats to compute acc')
    parser.add_argument('--loss', type=str, default='l2', choices=('l1', 'l2', 'huber'), help='Type of loss to use')

    # args for adaptively choosing which classes to continue trying
    parser.add_argument('--to_keep', nargs='+', type=int, required=True)
    parser.add_argument('--n_samples', nargs='+', type=int, required=True)

    # optional arg for using DDIM
    parser.add_argument('--scheduler', type=str, choices=['EDM', 'DDIM'], default='DDIM')

    args = parser.parse_args()
    assert len(args.to_keep) == len(args.n_samples)
    

    # make run output folder
    name = f"v{args.version}_{args.n_trials}trials_"
    name += '_'.join(map(str, args.to_keep)) + 'keep_'
    name += '_'.join(map(str, args.n_samples)) + 'samples'
    if args.interpolation != 'bicubic':
        name += f'_{args.interpolation}'
    if args.loss == 'l1':
        name += '_l1'
    elif args.loss == 'huber':
        name += '_huber'
    if args.img_size != 512:
        name += f'_{args.img_size}'
    if args.extra is not None:
        run_folder = osp.join(LOG_DIR, args.dataset + '_' + args.extra, name)
    else:
        run_folder = osp.join(LOG_DIR, args.dataset, name)
    os.makedirs(run_folder, exist_ok=True)
    print(f'Run folder: {run_folder}')

    # set up dataset and prompts
    interpolation = INTERPOLATIONS[args.interpolation]
    transform = get_transform(interpolation, args.img_size)
    latent_size = args.img_size // 8
    target_dataset = get_target_dataset(args.dataset, train=args.split == 'train', transform=transform)
    prompts_df = pd.read_csv(args.prompt_path)

    # load pretrained models
    vae, tokenizer, text_encoder, unet, scheduler = get_sd_model(args)
    vae = vae.to(device)
    text_encoder = text_encoder.to(device)
    unet = unet.to(device)
    torch.backends.cudnn.benchmark = True
    
    inverter = NullInversion(unet, vae, tokenizer, text_encoder, scheduler)

    # load noise
    if args.noise_path is not None:
        all_noise = torch.load(args.noise_path).to(device)
        print('Loaded noise from', args.noise_path)
    else:
        all_noise = None

    # refer to https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L276
    # prompts = prompts_df.prompt.tolist()
    
    prompts = prompts_df.prompt.tolist()
    prompts[-1] = ""
    
    text_input = tokenizer(prompts, padding="max_length",
                           max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    embeddings = []
    with torch.inference_mode():
        for i in range(0, len(text_input.input_ids), 100):
            text_embeddings = text_encoder(
                text_input.input_ids[i: i + 100].to(device),
            )[0]
            embeddings.append(text_embeddings)
    text_embeddings = torch.cat(embeddings, dim=0)
    assert len(text_embeddings) == len(prompts_df)

    # subset of dataset to evaluate
    if args.subset_path is not None:
        idxs = np.load(args.subset_path).tolist()
    else:
        idxs = list(range(len(target_dataset)))
    idxs_to_eval = idxs[args.worker_idx::args.n_workers]

    formatstr = get_formatstr(len(target_dataset) - 1)
    correct = 0
    total = 0
    pbar = tqdm.tqdm(idxs_to_eval)
    for i in pbar:
        if total > 0:
            pbar.set_description(f'Acc: {100 * correct / total:.2f}%')
        fname = osp.join(run_folder, formatstr.format(i) + '.pt')
        if os.path.exists(fname):
            print('Skipping', i)
            if args.load_stats:
                data = torch.load(fname)
                correct += int(data['pred'] == data['label'])
                total += 1
            continue
        image, label = target_dataset[i]
        with torch.no_grad():
            img_input = image.to(device).unsqueeze(0)
            if args.dtype == 'float16':
                img_input = img_input.half()
            x0 = vae.encode(img_input).latent_dist.mean
            x0 *= 0.18215
        pred_idx, pred_errors = eval_prob_adaptive_nti(unet, x0, text_embeddings, scheduler, inverter, args, latent_size, all_noise)
        pred = prompts_df.classidx[pred_idx]
        
        print(f'Label: {label}, Pred: {pred}')
        
        torch.save(dict(errors=pred_errors, pred=pred, label=label), fname)
        if pred == label:
            correct += 1
        total += 1
        
        print(f'Acc: {100 * correct / total:.2f}%')


if __name__ == '__main__':
    main()
