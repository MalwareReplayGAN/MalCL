import torch

def configurate(args, config):

    if (args.final_classes - args.init_classes) % args.n_inc != 0:
        print("error")

    num_task = int(((args.final_classes - args.init_classes) / args.n_inc) + 1)
    config.nb_task = num_task

    argu = vars(args)
    for var in argu:
        config.set(var, argu[var])

    if args.use_cuda : config.device = 'cuda'
    else : config.device = "cpu"
    
    config.ls_a = []
    config.z_dim = 62
    


def torch_setting(config):
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"number of available GPUs: {device_count}")
    else:
        print("can't use GPU")

    torch.cuda.empty_cache()

