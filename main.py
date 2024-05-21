import torch
import argparse
import time
from parsers.parser import Parser
from parsers.config import get_config
from Trainer import Trainer
from Sampler import Sampler_mol,Sampler

# def main(type_args):
#     ts = time.strftime('%b%d-%H:%M:%S', time.gmtime())
#     args=Parser().parse()
#     config=get_config(args.config,args.seed)
#
#     #----------Train----------
#     if type_args.type=="train":
#         trainer=Trainer(config)
#         ckpt=trainer.train(ts)
#         if "sample" in config.keys():
#             config.ckpt=ckpt
#             sampler=Sampler(config)
#             sampler.sample()
#
#     #----------Generation----------
#     elif type_args.type=="sample":
#         if config.data.data in ['QM9', 'ZINC250k']:
#             sampler=Sampler_mol(config)
#         else:
#             sampler=Sampler(config)
#
#     else:
#         raise ValueError(f'Wrong type : {type_args.type}')
#
# if __name__=="__main__":
#
#     type_args=argparse.ArgumentParser()
#     type_args.add_argument("--type",type=str,required=True)
#     main(type_args.parse_known_args()[0])


class Args:
    def __init__(self, work_type,config,seed):
        self.type = work_type
        self.config=config
        self.seed=seed


def main(work_type_args):
    ts = time.strftime('%b%d-%H:%M:%S', time.gmtime())
    # args = Parser().parse()
    args=work_type_args
    config = get_config(args.config, args.seed)

    # -------- Train --------
    if work_type_args.type == 'train':
        trainer = Trainer(config)
        ckpt = trainer.train(ts)
        if 'sample' in config.keys():
            config.ckpt = ckpt
            sampler = Sampler(config)
            sampler.sample()

    # -------- Generation --------
    elif work_type_args.type == 'sample':
        if config.data.data in ['QM9', 'ZINC250k']:
            sampler = Sampler_mol(config)
        else:
            sampler = Sampler(config)
        sampler.sample()

    else:
        raise ValueError(f'Wrong type : {work_type_args.type}')


if __name__ == '__main__':
    work_type_args = Args('train',"QM9",42)  # Change this to 'sample' if needed
    main(work_type_args)