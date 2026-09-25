import json
import platform
import random
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from .data import corrupt, digest, encode_texts, padded, read_pairs, train_tokenizer
from .model import Autoencoder, Config, PAD


def device_for(name):
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    torch.save(value, temp)
    temp.replace(path)


def load_model(path, device):
    # Only load trusted experiment checkpoints. Metadata consists of basic types.
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    model = Autoencoder(Config(**ckpt['config'])).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    tokenizer = Tokenizer.from_str(ckpt['tokenizer'])
    return model, tokenizer, ckpt


def amp_context(device, enabled):
    return torch.autocast('cuda', dtype=torch.bfloat16) if enabled else nullcontext()


@torch.no_grad()
def validation(model, rows, device, batch_size):
    model.eval()
    total, count = 0., 0
    # Local RNG isolation makes validation masking identical at every epoch.
    devices = [device.index or 0] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(12345)
        for start in range(0, len(rows), batch_size):
            ids = padded(rows[start:start+batch_size], device)
            logits = model(corrupt(ids), ids[:,:-1])
            targets = ids[:,1:]
            total += F.cross_entropy(logits.flatten(0,1), targets.flatten(),
                                     ignore_index=PAD, reduction='sum').item()
            count += targets.ne(PAD).sum().item()
    return total / count


def train(args):
    device = device_for(args.device)
    seed_all(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train_path = Path(args.data)/f'train.{args.lang}.txt'
    val_path = Path(args.data)/'val.jsonl'
    texts = [line.strip() for line in train_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    val_texts = [r[args.lang] for r in read_pairs(val_path)]
    if not texts or not val_texts:
        raise ValueError('Training and validation data must both be nonempty.')
    fingerprints = {'train':digest(train_path), 'val':digest(val_path)}
    config_args = dict(width=args.width, heads=args.heads, layers=args.layers,
                       slots=args.slots, max_len=args.max_len, dropout=args.dropout)
    resumed = None
    if args.resume:
        resumed = torch.load(out/'last.pt', map_location='cpu', weights_only=True)
        if resumed['data_hashes'] != fingerprints or resumed['lang'] != args.lang:
            raise ValueError('Resume requires the same language and unchanged training/validation files.')
        if any(resumed['config'][k] != v for k,v in config_args.items()):
            raise ValueError('Resume architecture differs. Repeat the original architecture flags.')
        for key in ['seed','batch_size','lr','mask_rate']:
            if resumed['train_args'][key] != getattr(args,key):
                raise ValueError(f'Resume requires the original --{key.replace("_", "-")}.')
        tokenizer = Tokenizer.from_str(resumed['tokenizer'])
    else:
        if any(out.iterdir()):
            raise ValueError('Run directory is not empty. Use --resume or a new --out.')
        if args.tokenizer:
            tokenizer = Tokenizer.from_file(args.tokenizer)
            tokenizer.save(str(out/'tokenizer.json'))
        else:
            tokenizer = train_tokenizer(out/'tokenizer.json', texts, args.vocab_size)
    tokenizer.save(str(out/'tokenizer.json'))
    config = Config(vocab_size=tokenizer.get_vocab_size(), **config_args)
    model = Autoencoder(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    start_epoch, best = 0, float('inf')
    use_amp = device.type == 'cuda' and torch.cuda.is_bf16_supported() and not args.fp32
    if resumed:
        model.load_state_dict(resumed['model'])
        optimizer.load_state_dict(resumed['optimizer'])
        start_epoch, best = resumed['epoch'], resumed['best_val']
        torch.set_rng_state(resumed['rng_cpu'])
        if device.type == 'cuda' and resumed['rng_cuda']:
            torch.cuda.set_rng_state_all(resumed['rng_cuda'])
    rows = encode_texts(tokenizer, texts, config.max_len)
    val_rows = encode_texts(tokenizer, val_texts, config.max_len)
    manifest = {'python':platform.python_version(),'torch':str(torch.__version__),
                'device':str(device),'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None,
                'bf16':use_amp,'config':asdict(config),'data_hashes':fingerprints,'args':vars(args)}
    (out/'environment.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'parameters':sum(p.numel() for p in model.parameters()),
                      'device':str(device),'bf16':use_amp,'start_epoch':start_epoch}), flush=True)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(args.seed+epoch)).tolist()
        loss_sum, tokens = 0., 0
        for step, start in enumerate(range(0,len(rows),args.batch_size)):
            ids = padded([rows[i] for i in order[start:start+args.batch_size]], device)
            source = corrupt(ids,args.mask_rate)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(device,use_amp):
                logits = model(source, ids[:,:-1])
                loss = F.cross_entropy(logits.flatten(0,1),ids[:,1:].flatten(),ignore_index=PAD)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            n = ids[:,1:].ne(PAD).sum().item()
            loss_sum += loss.item()*n
            tokens += n
            if step % 50 == 0:
                print(f'epoch={epoch+1} batch={step} loss={loss.item():.4f}',flush=True)
        val = validation(model,val_rows,device,args.batch_size)
        improved = val < best
        best = min(best,val)
        checkpoint = {'format':1,'model':model.state_dict(),'config':asdict(config),
                      'tokenizer':tokenizer.to_str(),'lang':args.lang,'epoch':epoch+1,
                      'best_val':best,'optimizer':optimizer.state_dict(),
                      'data_hashes':fingerprints,'train_args':vars(args),
                      'rng_cpu':torch.get_rng_state(),
                      'rng_cuda':torch.cuda.get_rng_state_all() if device.type=='cuda' else []}
        atomic_save(checkpoint,out/'last.pt')
        if improved:
            atomic_save(checkpoint,out/'best.pt')
        metrics = {'epoch':epoch+1,'train_loss':loss_sum/tokens,'val_masked_loss':val}
        with (out/'metrics.jsonl').open('a') as f:
            f.write(json.dumps(metrics)+'\n')
        print(json.dumps(metrics),flush=True)


@torch.no_grad()
def collect_latents(model, tokenizer, texts, device, batch_size):
    rows = encode_texts(tokenizer,texts,model.config.max_len)
    return torch.cat([model.encode(padded(rows[i:i+batch_size],device)).float().cpu()
                      for i in range(0,len(rows),batch_size)])


def evaluate(args):
    from sacrebleu.metrics import CHRF
    device = device_for(args.device)
    source,st,sc = load_model(args.source,device)
    target,tt,tc = load_model(args.target,device)
    if (source.config.slots,source.config.width)!=(target.config.slots,target.config.width):
        raise ValueError('Source and target latent shapes differ.')
    pairs = read_pairs(args.pairs)
    if args.limit:
        pairs = pairs[:args.limit]
    if len(pairs)<2:
        raise ValueError('Evaluation needs at least two examples.')
    src_text = [r[sc['lang']] for r in pairs]
    references = [r[tc['lang']] for r in pairs]
    z = collect_latents(source,st,src_text,device,args.batch_size)
    if args.shuffle_latents:
        # A cyclic shift changes every source association, even for small tests.
        z = z.roll(1,0)
    outputs = []
    with torch.no_grad():
        for i in range(0,len(z),args.batch_size):
            generated = target.generate(z[i:i+args.batch_size].to(device))
            outputs.extend(tt.decode_batch(generated.cpu().tolist(),skip_special_tokens=True))
    metric = CHRF()
    score = metric.corpus_score(outputs,[references])
    result = {'n':len(pairs),'chrf':score.score,'chrf_signature':str(metric.get_signature()),
              'exact_match':sum(a.strip()==b.strip() for a,b in zip(outputs,references))/len(pairs),
              'source':args.source,'target':args.target,'mode':'direct',
              'shuffle_latents':args.shuffle_latents,'pairs_sha256':digest(args.pairs)}
    out = Path(args.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2))
    with out.with_suffix('.predictions.jsonl').open('w',encoding='utf-8') as f:
        for src,ref,pred in zip(src_text,references,outputs):
            f.write(json.dumps({'source':src,'reference':ref,'prediction':pred},ensure_ascii=False)+'\n')
    print(json.dumps(result,indent=2))


def export(args):
    model,_,ckpt = load_model(args.checkpoint,'cpu')
    out = Path(args.out)
    metadata = {k:ckpt[k] for k in ['config','tokenizer','lang','format']}
    atomic_save({**metadata,'state_dict':model.encoder_state(),'component':'encoder'},out/'encoder.pt')
    atomic_save({**metadata,'state_dict':model.decoder_state(),'component':'decoder'},out/'decoder.pt')
    print(f'Exported encoder.pt and decoder.pt to {out}')
