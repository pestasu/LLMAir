import argparse
import os
import json
import random
import numpy as np
import torch
import torch.distributed as dist
from pathlib import Path
from utils.tools import set_logger, serializable_parts_of_dict, gen_version
from config.config import get_config 
from exp import *
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def main():
    fix_seed = [1111,2222,3333,4444,5555,6666,7777,8888,9999]
    random.seed(fix_seed[0])
    torch.manual_seed(fix_seed[0])
    np.random.seed(fix_seed[0])

    parser = argparse.ArgumentParser(description='Air quality Prediction Based on Pretrained LLM')

    # basic config
    parser.add_argument('--data', type=str, required=True, default='beijing', help='dataset name')
    parser.add_argument('--model', type=str, required=True, default='llmair',
                        help='model name, options: [llmair, airformer, gagnn...]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--num_workers', type=int, default=4, help='data loader num workers') 
    parser.add_argument('--n_exp', type=int, default=1, help='experiments times')
    parser.add_argument('--version', type=int, default=-1, help='experiments version')

    # pt
    parser.add_argument('--need_pt', type=int, default=0, help='whether continue pretrain')

    # optimization
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--save_iter', type=int, default=400, help='save model each save_iter')
    parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate: consine, type1, type2, type3')
    parser.add_argument('--max_grad_norm', type=float, default=5.0)
    parser.add_argument('--patience', type=int, default=15)

    # GPU
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    args = parser.parse_args()

    config = get_config(args.model, args.data)
    config.update(vars(args))

    # save params and loggers
    setting = '{}_{}_{}_{}_bs{}_lr{}_wd{}'.format(
                config.model,
                config.data,
                config.seq_len,
                config.pred_len,
                config.batch_size,
                config.learning_rate,
                config.weight_decay
                )
    if config.is_training:
        config['version'] = gen_version(config)

    save_floder = os.path.join('saved/'+'/'+config.data+'/'+config.model, f"{setting}_{config.version}")
    data_floder = os.path.join(config.data_root_path, config.data_path)
    pt_dir = Path(save_floder) / "pt"
    log_dir = Path(save_floder) / "logs"
    Path(pt_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    config.pt_dir = pt_dir
    config.data_floder = data_floder
    logger = set_logger(log_dir, config.model, config.data, verbose_level=1)
    serializable_dict = serializable_parts_of_dict(config)
    logger.info(json.dumps(serializable_dict, indent=4))
    logger.info(f'>>>>>>>>>>>>>>>>>>>>>>>>>> [ {config.model}-{config.data}({config.version}) ]<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

    import exp
    exp_name = f'Exp_{args.model}'
    Exp = getattr(exp, exp_name)

    if config.is_training:
        for ii in range(config.n_exp):
            # setting record of experiments
            exp = Exp(config, ii)  # set experiments
            # if (args.use_multi_gpu and args.local_rank == 0) or not args.use_multi_gpu:
            #     logger.info('>>>>>>>>>>>>>>>>>>>>>>>>>>start training : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.train(setting)
            if (args.use_multi_gpu and args.local_rank == 0) or not args.use_multi_gpu:
                print('>>>>>>>>>>>>>>>>>>>>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()

    else:
        ii = 0
        setting = '{}_{}_{}_{}_bs{}_lr{}_wd{}'.format(
            config.model,
            config.data,
            config.seq_len,
            config.pred_len,
            config.batch_size,
            config.learning_rate,
            config.weight_decay
            )
        exp = Exp(config, ii)  # set experiments
        exp.test(setting, is_test=True)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()