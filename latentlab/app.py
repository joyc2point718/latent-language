"""Named monolingual models and self-contained encoder/decoder combinations."""
import argparse
import json
import re
import secrets
from pathlib import Path
from types import SimpleNamespace
import torch
from tokenizers import Tokenizer
from .data import make_data, encode_texts, padded, digest
from .experiment import train as train_autoencoder, load_model, atomic_save, device_for
from .model import Autoencoder, Config

LANGUAGES = {'en':'en', 'english':'en', 'es':'es', 'spanish':'es'}
DEFAULTS = dict(width=256, heads=4, layers=4, slots=8, max_len=64, dropout=0.1,
                vocab_size=8000, lr=0.0003, mask_rate=0.3, fp32=False, tokenizer=None)


def model_dir(root, name):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', name):
        raise ValueError('Model names must contain only letters, numbers, underscores or hyphens.')
    return Path(root).resolve()/name


def train_named(language, name, models_dir='models', data=None, epochs=None,
                batch_size=None, seed=None, resume=False, device='auto'):
    lang = LANGUAGES.get(language.lower())
    if lang is None:
        raise ValueError('Supported languages: english/en and spanish/es.')
    out = model_dir(models_dir,name)
    if resume:
        saved = torch.load(out/'last.pt',map_location='cpu',weights_only=True)
        if saved['lang'] != lang:
            raise ValueError('The saved model was trained in a different language.')
        options = saved['train_args'].copy()
        options.update(out=str(out),lang=lang,device=device,resume=True)
        if data is not None: options['data']=str(Path(data).resolve())
        for key,value in [('epochs',epochs),('batch_size',batch_size),('seed',seed)]:
            if value is not None: options[key]=value
    else:
        if out.exists():
            raise ValueError(f'Model name already exists: {name}. Use --resume or a new name.')
        data_path=Path(data).resolve() if data else Path('data/toy').resolve()
        if data is None and not data_path.exists():
            print('Generating the controlled toy corpus (not general English/Spanish text).',flush=True)
            make_data(data_path)
        options = dict(DEFAULTS, data=str(data_path),out=str(out),lang=lang,device=device,
                       epochs=15 if epochs is None else epochs,
                       batch_size=128 if batch_size is None else batch_size,
                       seed=secrets.randbelow(2**31) if seed is None else seed,resume=False)
    if options['epochs']<=0 or options['batch_size']<=0:
        raise ValueError('Epochs and batch size must be positive.')
    train_autoencoder(SimpleNamespace(**options))
    print(f'Saved model "{name}" ({lang}) in {out}')


def merge_named(encoder, decoder, name, models_dir='models'):
    out=model_dir(models_dir,name)
    if out.exists():
        raise ValueError(f'Model name already exists: {name}. Choose a new name.')
    src_path=model_dir(models_dir,encoder)/'best.pt'
    tgt_path=model_dir(models_dir,decoder)/'best.pt'
    src,_,sc=load_model(src_path,'cpu')
    tgt,_,tc=load_model(tgt_path,'cpu')
    if (src.config.slots,src.config.width)!=(tgt.config.slots,tgt.config.width):
        raise ValueError('Encoder and decoder must have matching latent slot counts and widths.')
    artifact = {'format':'latentlab-merged-v1','name':name,
        'encoder':{'name':encoder,'lang':sc['lang'],'config':sc['config'],
                   'tokenizer':sc['tokenizer'],'weights':src.encoder_state(),'source_sha256':digest(src_path)},
        'decoder':{'name':decoder,'lang':tc['lang'],'config':tc['config'],
                   'tokenizer':tc['tokenizer'],'weights':tgt.decoder_state(),'source_sha256':digest(tgt_path)}}
    atomic_save(artifact,out/'merged.pt')
    print(f'Saved "{name}": {encoder} encoder ({sc["lang"]}) → {decoder} decoder ({tc["lang"]})')
    return out/'merged.pt'


class MergedModel:
    def __init__(self,path,device='auto'):
        self.device=device_for(device)
        artifact=torch.load(path,map_location='cpu',weights_only=True)
        if artifact.get('format')!='latentlab-merged-v1':
            raise ValueError('Expected a saved merged model.')
        components=[]
        for kind in ['encoder','decoder']:
            part=artifact[kind]
            model=Autoencoder(Config(**part['config']))
            expected=set(model.encoder_state() if kind=='encoder' else model.decoder_state())
            if set(part['weights'])!=expected:
                raise ValueError(f'Incomplete or unexpected {kind} weights.')
            model.load_state_dict(part['weights'],strict=False)
            # Remove unused modules after loading so inference keeps only selected components.
            unused=['decoder','output'] if kind=='encoder' else ['encoder','queries','pool','latent_norm']
            for attr in unused: delattr(model,attr)
            components.append(model.to(self.device).eval())
        self.encoder,self.decoder=components
        self.input_tokenizer=Tokenizer.from_str(artifact['encoder']['tokenizer'])
        self.output_tokenizer=Tokenizer.from_str(artifact['decoder']['tokenizer'])

    @torch.no_grad()
    def generate(self,text):
        rows=encode_texts(self.input_tokenizer,[text],self.encoder.config.max_len)
        z=self.encoder.encode(padded(rows,self.device))
        ids=self.decoder.generate(z)
        return self.output_tokenizer.decode(ids[0].tolist(),skip_special_tokens=True)


def generate_named(name,text,models_dir='models',device='auto'):
    folder=model_dir(models_dir,name)
    if (folder/'merged.pt').exists():
        return MergedModel(folder/'merged.pt',device).generate(text)
    device=device_for(device)
    model,tokenizer,_=load_model(folder/'best.pt',device)
    ids=padded(encode_texts(tokenizer,[text],model.config.max_len),device)
    with torch.no_grad():
        output=model.generate(model.encode(ids))
    return tokenizer.decode(output[0].tolist(),skip_special_tokens=True)


def list_models(root):
    root=Path(root)
    if not root.exists():
        print('No saved models yet.'); return
    found=False
    for folder in sorted(root.iterdir()):
        if not folder.is_dir(): continue
        path=folder/'merged.pt'
        if path.exists():
            ck=torch.load(path,map_location='cpu',weights_only=True)
            print(f'{folder.name}: {ck["encoder"]["name"]} ({ck["encoder"]["lang"]}) encoder → '
                  f'{ck["decoder"]["name"]} ({ck["decoder"]["lang"]}) decoder')
            found=True
        elif (folder/'best.pt').exists():
            ck=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
            print(f'{folder.name}: {ck["lang"]} autoencoder (best epoch {ck["epoch"]})')
            found=True
    if not found: print('No completed models yet.')


def main():
    p=argparse.ArgumentParser(description='Train a named language model or combine saved encoders and decoders.')
    p.add_argument('--models-dir',default='models',help='Where trained and merged models are stored')
    sub=p.add_subparsers(dest='action',required=True)
    t=sub.add_parser('train')
    t.add_argument('--language',choices=list(LANGUAGES),required=True)
    t.add_argument('--name',required=True)
    t.add_argument('--data',help='Custom training/validation data directory; default generates toy data')
    t.add_argument('--epochs',type=int)
    t.add_argument('--batch-size',type=int)
    t.add_argument('--seed',type=int)
    t.add_argument('--resume',action='store_true')
    t.add_argument('--device',default='auto')
    m=sub.add_parser('merge')
    m.add_argument('--encoder',required=True,help='Name of the trained model supplying the encoder')
    m.add_argument('--decoder',required=True,help='Name of the trained model supplying the decoder')
    m.add_argument('--name',required=True,help='Name for the saved merged model')
    g=sub.add_parser('generate')
    g.add_argument('--model',required=True)
    g.add_argument('--text',required=True)
    g.add_argument('--device',default='auto')
    sub.add_parser('list')
    args=vars(p.parse_args());action=args.pop('action')
    if action=='train': train_named(**args)
    elif action=='merge': merge_named(**args)
    elif action=='generate':
        args['name']=args.pop('model')
        print(generate_named(**args))
    else: list_models(args['models_dir'])

if __name__=='__main__': main()
