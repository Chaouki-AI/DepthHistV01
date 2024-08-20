# This code was written by Mohammed Chaouki ZIARA, 
# For Ph.D. Project @ RCAM Laboratory, Djilali Liabes University - Algeria 
# bellow is the main code 
# please Modify the args on the args_****.txt files 
# Contact: medchaoukiziara@gmail.com || me@chaouki.pro

import argparse
from helper import load_model, load_images_deps, load_dl, \
                    load_optimizer_and_scheduler, load_devices, save_ckpt, \
                    pick_n_elements, trainer, evaluator, save_metrics_to_file, load_summary_writer



if __name__ == '__main__':
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    # Model's Specifications
    parser.add_argument("--simple", action="store_false", 
                        help="Choose whether using a model with histogram layer or only Encoder-Decoder model") 
    
    parser.add_argument("--bins", default=128, type=int,
                        help='number of bins to be used in histogram layer')
    parser.add_argument("--bins-loss", default=128, type=int,
                        help='number of bins to be used in loss')
    parser.add_argument("--backbone", default='efficientnet', choices=['efficientnet', 'ResNet'],
                         help='backbone model to be used in the model')  

    # Training Hyperparameters
    parser.add_argument("--epochs", default=100, type=int,
                        help='number of epochs')
    parser.add_argument("--bs", default=2, type=int,
                        help='batch size for training')
    parser.add_argument("--all-images", action="store_false", default=True,
                        help="to specify if we will use all the images for training or only a limited number of images")
    parser.add_argument("--Nb-imgs", type=int, default=2000, 
                        help='Number of images to be used for training (required if use-all-images is true)')

    # Dataset parameters 
    parser.add_argument("--dataset", default="kitti", type=str, 
                        help='dataset used for training, kitti, nyu or Daad')
    parser.add_argument("--train-txt", default="./Data/splits/kitti/kitti_eigen_train_files_with_gt.txt", type=str, 
                        help='path to the filenames text file for training')
    parser.add_argument("--test-txt", default="./Data/splits/kitti/kitti_eigen_test_files_with_gt.txt", type=str, 
                        help='path to the filenames text file for testing')
    parser.add_argument("--images-path", default="./Data/images/", type=str, 
                        help='path of images')
    parser.add_argument("--depths-path", default="./Data/depths/", type=str, 
                        help='path of depths')
    parser.add_argument("--max_depth", type=float, default=80., 
                        help='maximum depth in estimation')
    parser.add_argument("--min_depth", type=float, default=1e-3,
                        help='minimum depth in estimation')
    parser.add_argument("--image_height", type=int, default=352, 
                        help='image input height')
    parser.add_argument("--image_width", type=int, default=704,
                        help='image input width')

    # Saving parameters 
    parser.add_argument("--ckpt", default="./checkpoints/", type=str, 
                        help='path used to save checkpoints')
    parser.add_argument("--txts", default="./txts/", type=str, 
                        help='path used to write and monitor metrics')
    parser.add_argument("--name", default="Model Name", 
                        help='This name will be used to save the model')

    # Loss parameters
    parser.add_argument("--scale-silog", type=float, default=5., 
                        help='factor to be multiplied with silog loss')
    parser.add_argument("--scale-joint", type=float, default=1., 
                        help='factor to be multiplied with joint loss')
    parser.add_argument("--scale-hist", type=float, default=0.1, 
                        help='factor to be multiplied with histogram loss')
    parser.add_argument("--t", type=float, default=0.001, 
                        help='factor to be used as regulizer')

    # Optimizer parameters
    parser.add_argument("--optimizer", default="AdamW", type=str, help="name of the optimizer to be used for training",
                        choices=['AdamW', 'RMSprop'])
    parser.add_argument("--lr", "--learning-rate", default=0.0007, type=float, 
                        help='max learning rate')
    parser.add_argument("--lr-pol", "--lr-pol", default='OCL', type=str, 
                        help='learning rate policy')
    parser.add_argument("--wd", "--weight-decay", default=0.1, type=float, 
                        help='weight decay')
    parser.add_argument("--same-lr", default=True, action="store_true",
                        help="Use same LR for all params")
    
    #GPU parameters
    parser.add_argument("--gpu-tr", default=True, action="store_true",
                        help="Use the GPU or Not for training")
    parser.add_argument("--gpu-ts", default=False, action="store_true",
                        help="Use the GPU or Not for testing and evaluation")

    parser.add_argument("--crop", default="eigen", type=str, help="type of the Crop to be performed for evaluation",
                        choices=['eigen', 'garg'])

    args = parser.parse_args()
    
    #Load Summary Writer 
    writer = load_summary_writer(args)


    #Load The model 
    model = load_model(args)

    #Create The dataloader 
    
    imgs_tr, deps_tr = load_images_deps(args, train=True)
    imgs_ts, deps_ts = load_images_deps(args, train=False)
    valid_dl = load_dl(args, imgs=imgs_ts[:], depths=deps_ts[:], train=False)
    print(f"\n\n\ndata from {args.dataset} dataset loaded Perfectly with : \
          \n{len(imgs_tr)} images for training, \
          \n{len(imgs_ts)} for Testing\n\n\n" )

    #Create the Optimizer 
    optimizer, scheduler = load_optimizer_and_scheduler(args, model, N_imgs = len(imgs_tr))

    #Get the Devices ; 
    device_tr, device_ts = load_devices(args)
    filename = None
 
    #loop
    for epoch in range(0, args.epochs):
        
        images, depths = pick_n_elements(args, imgs_tr, deps_tr )
        
        #create Train Dataloader
        train_dl = load_dl(args, imgs=images, depths=depths, train=True)
        #train
        model, optimizer, scheduler, writer = trainer(args, model, train_dl, optimizer, scheduler, epoch, device_tr, writer)
        #evaluate
        model, filename, writer, metrics =  evaluator(args, model, valid_dl, epoch, device = device_ts, writer = writer, filename=filename)
        #save the ckpts
        save_ckpt(args, model, metrics)
     

