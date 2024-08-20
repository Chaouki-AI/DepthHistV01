# This Code was written by Mohammed Chaouki ZIARA, 
# For Ph.D. Project @ RCAM Laboratory, Djilali Liabes University - Algeria 
# Below is the code for a functions used to build, train, evaluate model and for results visualization
# Contact: medchaoukiziara@gmail.com || me@chaouki.pro

from tools.loss import SILogLoss, Hist1D_loss, Hist2D_loss
from torch.utils.tensorboard import SummaryWriter
from Models.model import DepthHist
from tools.dataloader import DepthDataLoader
from tools.evaluation import *
from datetime import datetime
from tqdm import tqdm
import numpy as np
import random
import torch
import os 

def load_model(args):
    model = DepthHist.build(bins=args.bins, simple=args.simple, backbone=args.backbone)
    print(f"The total Number of trainable parameters will be : {count_parameters(model)}")
    return model

def data_extractor_kitti(args, train = True):

    file  = open(args.train_txt) if train else open(args.test_txt)
    lines = file.readlines()
    file.close()
    path_tr_img = f"{args.images_path}train/"
    path_tr_dep = f"{args.depths_path}train/"
    
    path_ts_img = f"{args.images_path}val/"
    path_ts_dep = f"{args.depths_path}val/"
    
    
    imgs = []
    deps = []
    
    for i in lines :
        image, depth = i.strip().split(' ')[:-1]
        image = f"{image.split('/')[1]}/{image}" 
        if os.path.isfile(path_tr_img+image) and os.path.isfile(path_tr_dep+depth):
            imgs.append(path_tr_img+image)
            deps.append(path_tr_dep+depth)

        elif os.path.isfile(path_ts_img+image) and os.path.isfile(path_ts_dep+depth):
            imgs.append(path_ts_img+image)
            deps.append(path_ts_dep+depth)
        else : 
            pass 
            #print(f"{image} or {depth} doesn't exist")
            
    return imgs, deps

def data_extractor_nyu(args , train = True):

    file  = open(args.train_txt) if train else open(args.test_txt)
    lines = file.readlines()
    file.close()
    
    imgs = []
    deps = []
    
    for i in lines :
        image, depth = i.strip().split(' ')[:-1]
        image_file = args.images_path+image if train else  args.images_path+'/'+image
        depth_file = args.depths_path+depth if train else  args.depths_path+'/'+depth
        if os.path.isfile(image_file) and os.path.isfile(depth_file):
            imgs.append(image_file)
            deps.append(depth_file)
        else : 
            print(args.images_path+image , os.path.isfile(depth_file))

    return imgs, deps

def load_images_deps(args, train = True):
    if args.dataset == 'kitti':
        return data_extractor_kitti(args=args, train=train)
    elif args.dataset == 'nyu':
        return data_extractor_nyu(args=args, train=train)
    
def load_dl(args, imgs, depths, train = True):
    dataloader = DepthDataLoader(args, imgs, depths, mode = 'train' if train else 'online_eval').data
    return dataloader

def create_lr_lambda(total_steps):
    def lr_lambda(current_step):
        return 1 - (current_step / total_steps) * 0.99  # Gradually reduce to lr / 100
    return lr_lambda

def load_optimizer_and_scheduler(args, model, N_imgs):
    optimizer_name = args.optimizer
    optimizer_class = getattr(torch.optim, optimizer_name)
    params = [{"params": model.get_1x_lr_params() , "lr": args.lr/10},
              {"params": model.get_10x_lr_params(), "lr": args.lr}]
    
    if  args.all_images :
        
        steps_per_epoch = int(N_imgs/args.bs) if N_imgs % args.bs == 0 else int(N_imgs/args.bs) + 1
        print(f'iterations = {N_imgs/args.bs}')
    else :
        steps_per_epoch = int(args.Nb_imgs/args.bs)

    optimizer = optimizer_class(params, weight_decay=args.wd, lr=args.lr)
    if args.lr_pol == 'OCL' :
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.lr, epochs=args.epochs, 
                                                        steps_per_epoch= steps_per_epoch,
                                                        cycle_momentum=True, three_phase = False,
                                                        base_momentum=0.85, max_momentum=0.95, last_epoch=-1,
                                                        div_factor = args.epochs*2.5, anneal_strategy = 'linear', 
                                                        final_div_factor = args.epochs/1)
    else : 
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, create_lr_lambda(steps_per_epoch*args.epochs))

    return optimizer, scheduler

def load_losses(args):
    Silog_loss = SILogLoss(args)
    Histo_loss = Hist1D_loss(args)
    Joint_loss = Hist2D_loss(args)

    return Silog_loss, Joint_loss, Histo_loss

def pick_n_elements(args, list1, list2):
    if not args.all_images:
        # Ensure both lists have the same length
        assert len(list1) == len(list2), "Both lists must have the same length"
        
        # Generate a list of indices
        indices = list(range(len(list1)))
        
        # Randomly sample N indices from the list of indices
        selected_indices = random.sample(indices, args.Nb_imgs)
        
        # Use the selected indices to pick elements from both lists
        picked_list1 = [list1[i] for i in selected_indices]
        picked_list2 = [list2[i] for i in selected_indices]
        return picked_list1, picked_list2
    else :
        return list1, list2

def load_devices(args):
    if torch.cuda.is_available():
        if args.gpu_tr and args.gpu_ts :
            print("\nGPU Will Be used For training and For Testing \n")
            device_tr = torch.device("cuda")
            device_ts = torch.device("cuda")

        elif args.gpu_tr and not args.gpu_ts :
            print("\nGPU Will Be used For training and CPU For testing\n")
            device_tr = torch.device("cuda")
            device_ts = torch.device("cpu")

        elif args.gpu_ts and not args.gpu_tr :
            print("\nCPU Will Be used For training and GPU For testing\n")
            device_tr = torch.device("cpu")
            device_ts = torch.device("cuda")
    else :
        print("\nCPU Will Be used For training and CPU For testing\n")
        device_tr = torch.device("cpu")
        device_ts = torch.device("cpu")
    return device_tr, device_ts

def trainer(args, model, dataloader, optimizer, scheduler, epoch, device, writer):
    model.train()

    #Load Losses Functions
    Si_loss, Joint_loss, Hist_loss = load_losses(args)
    #device, _ = get_device(args, epoch)
    
    model = model.to(device)
    progress_bar = tqdm(enumerate(dataloader), desc=f"Epoch: {epoch + 1}/{args.epochs}. Loop: Train", total=len(dataloader))
    for i, batch in progress_bar:
        optimizer.zero_grad()

        img = batch['image'].to(device)
        depth = batch['depth'].to(device)
        
        pred = model(img)

        mask_min = depth > args.min_depth
        mask_max = depth <= args.max_depth
        mask     = torch.logical_and(mask_min, mask_max)
        l_dense  = Si_loss(pred, depth, mask=mask.to(torch.bool), interpolate=True)
        l_histo  = Hist_loss(depth, pred, mask=mask.to(torch.bool), interpolate = True)
        j_histo  = Joint_loss(depth, pred, mask=mask.to(torch.bool), interpolate = True)

        loss = args.scale_silog * l_dense  + l_histo * args.scale_hist + j_histo * args.scale_joint  
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)  # optional
        optimizer.step()
        #step += 1
        scheduler.step()

        # Update the tqdm description with the current value of l_chamfer
        progress_bar.set_description(f"Epoch: {epoch + 1}/{args.epochs}. Loop: Train. l_dense: {l_dense.item():.4f} l_histo: {l_histo.item():.4f} j_histo: {j_histo.item():.4f}")

        if i % 5 == 0:
            iteration_number = epoch * len(dataloader) + i
            writer.add_scalar('Loss/Train', loss.item(), iteration_number)
            current_lr = scheduler.get_last_lr()[0]
            writer.add_scalar('LR', current_lr, iteration_number)
    return model, optimizer, scheduler, writer 

def __evaluator__(args, model, test_loader, epoch, range , device, writer):
    Si_loss, _, _ = load_losses(args)
    model.eval()


    model = model.to(device)
    with torch.no_grad():
        val_si = RunningAverage()
        metrics = RunningAverageDict()
        for batch in tqdm(test_loader, desc=f"Epoch: {epoch + 1}/{args.epochs} Validation for {int(range)} Meters"):
            img = batch['image'].to(device)
            depth = batch['depth'].to(device)
            
            
            depth = depth.squeeze().unsqueeze(0).unsqueeze(0)
 
            pred = model(img)

            mask = depth > args.min_depth

            l_dense = Si_loss(pred, depth, mask=mask.to(torch.bool), interpolate=True)
            val_si.append(l_dense.item())

            pred = torch.nn.functional.interpolate(pred, depth.shape[-2:], mode='bilinear', align_corners=True)

            pred = pred.squeeze().cpu().numpy()
            pred[pred < args.min_depth] = args.min_depth
            pred[pred > range] = range
            pred[np.isinf(pred)] = range
            pred[np.isnan(pred)] = args.min_depth

            gt_depth = depth.squeeze().cpu().numpy()
            valid_mask = np.logical_and(gt_depth > args.min_depth, gt_depth < range)

            

            if args.crop == "garg":
                #print('here')
                gt_height, gt_width = gt_depth.shape
                eval_mask = np.zeros(valid_mask.shape)
                eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

            elif args.crop == "eigen":
                gt_height, gt_width = gt_depth.shape
                eval_mask = np.zeros(valid_mask.shape)
                if args.dataset == 'kitti':
                    eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height),
                    int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
                elif args.dataset == 'nyu':
                    eval_mask[45:471, 41:601] = 1
            else :
                eval_mask = np.ones(valid_mask.shape)
            valid_mask = np.logical_and(valid_mask, eval_mask)
            metrics.update(compute_errors(gt_depth[valid_mask], pred[valid_mask]))
        
        values = {key: float(f"{value:.5f}") for key, value in metrics.get_value().items()}
        writer.add_scalar(f'Metrics {int(range)} Meters/rmse'    , values['rmse'] , epoch)
        writer.add_scalar(f'Metrics {int(range)} Meters/sq_rel'  , values['sq_rel'] , epoch)
        writer.add_scalar(f'Metrics {int(range)} Meters/rmse_log', values['rmse_log'] , epoch)
        writer.add_scalar(f'Metrics {int(range)} Meters/acc1'    , values['a1'] , epoch)
        writer.add_scalar(f'Metrics {int(range)} Meters/acc2'    , values['a2'] , epoch)
        writer.add_scalar(f'Metrics {int(range)} Meters/acc3'    , values['a3'] , epoch)
        writer.add_scalar(f'Metrics {int(range)} Meters/abs_rel' , values['abs_rel'] , epoch)

        return metrics.get_value(), val_si, writer

def evaluator(args, model, valid_dl, epoch , device, writer, filename = None):
    if args.dataset == 'kitti' :
        #evaluate for 80meters and save on the txt file  
        metrics_80, _ , writer = __evaluator__(args, model, valid_dl, epoch, range = 80., device = device, writer = writer)
        metrics_80 = {key: float(f"{value:.5f}") for key, value in metrics_80.items()}
        filename = save_metrics_to_file(args, metrics_80, epoch, 80., filename=filename)

        #evaluate for 60meters and save on the txt file  
        metrics_60, _, writer = __evaluator__(args, model, valid_dl, epoch, range = 60., device = device, writer = writer)
        metrics_60 = {key: float(f"{value:.5f}") for key, value in metrics_60.items()}
        filename = save_metrics_to_file(args, metrics_60, epoch, 60., filename=filename)
        return model, filename, writer, metrics_80
    
    elif args.dataset == 'nyu' :
        #evaluate for 10meters and save on the txt file
        metrics_10, _ , writer = __evaluator__(args, model, valid_dl, epoch, range = 10., device = device, writer = writer)
        metrics_10 = {key: float(f"{value:.5f}") for key, value in metrics_10.items()}
        filename = save_metrics_to_file(args, metrics_10, epoch, 10., filename=filename)
        return model, filename, writer, metrics_10

def save_metrics_to_file(args, metrics, epoch, range, filename):
    if filename is None :
        dets = '\n'.join([f"{arg} : {getattr(args, arg)}" for arg in vars(args)])
        now = datetime.now()
        filename = f"{args.txts}/{args.name} Created on {now.strftime('%m_%d_%Y_%H_%M_%S')}.txt"
        explain = f"{dets}\n"
        sep = str('*'*100)
        additional_info = f"{explain}{sep}\n\nMetrics for {args.name} Created on {now.strftime('%m/%d/%Y, %H:%M:%S')}\n\n"
        additional_info = f'{additional_info}Epoch:{epoch+1} for range {int(range)}'
    else : 
        additional_info =  f"Epoch:{epoch+1}/{args.epochs} - MaxDepth {int(range)}"

    with open(filename, 'a') as file:
        if additional_info:
            file.write(additional_info + "\n")
        for key, value in metrics.items():
            file.write(f"{key}: {value}\n")
        if range == 60. :
            file.write(f"{str('-'*100)}\n") 
        file.write("\n") 
    return filename

def save_ckpt(args, model, metrics):
    os.makedirs(f"{args.ckpt}{args.name}",  exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.ckpt, f"{args.name}", f"{args.name}_abs_rel-{metrics['abs_rel']}_A1-{metrics['a1']}_best.pt"))

def load_summary_writer(args):
    writer = SummaryWriter()
    return writer

def load_weights(args, model, path=None, device = 'cpu'):
    if device is None : 
        _, device = load_devices(args)
    else : 
        device = torch.device(device)
    try :
        model.load_state_dict(torch.load(path, map_location = device))
    except Exception as e :
        print(e)
    return model
