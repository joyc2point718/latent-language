import argparse
from .data import make_data
from .experiment import train,align,evaluate,export


def main():
    parser = argparse.ArgumentParser(description='Independent language autoencoders and latent swaps')
    sub = parser.add_subparsers(dest='command',required=True)
    p = sub.add_parser('prepare',help='Create a reproducible controlled English/Spanish corpus')
    p.add_argument('--out',default='data/toy')
    for name,default in [('train',20000),('val',1000),('align',1000),('test',1000),('ood',500),('seed',42)]:
        p.add_argument('--'+name,type=int,default=default)
    p = sub.add_parser('train')
    p.add_argument('--data',default='data/toy')
    p.add_argument('--lang',choices=['en','es'],required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--device',default='auto')
    p.add_argument('--tokenizer',help='Reuse a tokenizer trained ONLY on this language training split')
    for name,default in [('epochs',15),('batch-size',128),('seed',1),('vocab-size',8000),('width',256),('heads',4),('layers',4),('slots',8),('max-len',64)]:
        p.add_argument('--'+name,type=int,default=default)
    for name,default in [('lr',0.0003),('dropout',0.1),('mask-rate',0.3)]:
        p.add_argument('--'+name,type=float,default=default)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--fp32',action='store_true')
    p = sub.add_parser('align')
    for name in ['source','target','pairs','out']:
        p.add_argument('--'+name,required=True)
    p.add_argument('--kind',choices=['orthogonal','linear'],default='orthogonal')
    p.add_argument('--n',type=int,default=1000)
    p.add_argument('--ridge',type=float,default=0.01)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--device',default='auto')
    p.add_argument('--shuffle-pairs',action='store_true')
    p = sub.add_parser('evaluate')
    for name in ['source','target','pairs','out']:
        p.add_argument('--'+name,required=True)
    p.add_argument('--bridge')
    p.add_argument('--shuffle-latents',action='store_true')
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--limit',type=int,default=0,help='0 evaluates all examples')
    p.add_argument('--device',default='auto')
    p = sub.add_parser('export')
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--out',required=True)
    args = parser.parse_args()
    if args.command=='prepare':
        kw=vars(args).copy(); kw.pop('command'); make_data(**kw)
    else:
        {'train':train,'align':align,'evaluate':evaluate,'export':export}[args.command](args)

if __name__=='__main__':
    main()
