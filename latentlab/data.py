import hashlib
import itertools
import json
import random
from pathlib import Path
import torch
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from .model import PAD, BOS, EOS, MASK

SPECIAL = ['[PAD]', '[BOS]', '[EOS]', '[UNK]', '[MASK]']
NOUNS = [('dog','perro'), ('cat','gato'), ('rabbit','conejo'), ('horse','caballo'),
         ('bear','oso'), ('fox','zorro'), ('wolf','lobo'), ('mouse','ratón')]
COLORS = [('red','rojo'), ('blue','azul'), ('green','verde'), ('white','blanco'), ('black','negro')]
VERBS = [('chase','chases','chased','persigue','persiguió'),
         ('follow','follows','followed','sigue','siguió'),
         ('watch','watches','watched','observa','observó'),
         ('help','helps','helped','ayuda','ayudó'),
         ('find','finds','found','encuentra','encontró'),
         ('greet','greets','greeted','saluda','saludó')]

def render(event):
    a, p, v, ac, pc, past, neg = event
    verb = VERBS[v]
    en_v = (('did not ' if past else 'does not ') + verb[0]) if neg else verb[2 if past else 1]
    es_v = ('no ' if neg else '') + verb[4 if past else 3]
    en = f'The {COLORS[ac][0]} {NOUNS[a][0]} {en_v} the {COLORS[pc][0]} {NOUNS[p][0]}.'
    es = f'El {NOUNS[a][1]} {COLORS[ac][1]} {es_v} al {NOUNS[p][1]} {COLORS[pc][1]}.'
    return {'en': en, 'es': es, 'event': list(event)}

def make_data(out, train=20000, val=1000, align=1000, test=1000, ood=500, seed=42):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output directory is not empty; choose a new data directory.')
    rng = random.Random(seed)
    ordinary, heldout = [], []
    for e in itertools.product(range(8), range(8), range(6), range(5), range(5), range(2), range(2)):
        if e[0] == e[1]:
            continue
        (heldout if (e[0], e[2]) in {(0, 5), (5, 3)} else ordinary).append(e)
    rng.shuffle(ordinary)
    rng.shuffle(heldout)
    if train + val + align + test > len(ordinary) or ood > len(heldout):
        raise ValueError('Requested more unique events than the controlled grammar supports.')
    out.mkdir(parents=True, exist_ok=True)
    offset = 0
    for split, count in [('train',train),('val',val),('align',align),('test',test),('ood',ood)]:
        events = heldout[:count] if split == 'ood' else ordinary[offset:offset+count]
        if split != 'ood':
            offset += count
        rows = [render(e) for e in events]
        if split == 'train':
            # Each language has independent ordering and no pair IDs in its training file.
            for lang in ['en', 'es']:
                texts = [r[lang] for r in rows]
                rng.shuffle(texts)
                (out/f'train.{lang}.txt').write_text('\n'.join(texts)+'\n', encoding='utf-8')
        else:
            (out/f'{split}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    (out/'manifest.json').write_text(json.dumps({'seed':seed,'counts':dict(train=train,val=val,align=align,test=test,ood=ood),
        'ood_agent_verb':[['dog','greet'],['fox','help']], 'grammar':'controlled-v1'},indent=2))


def read_pairs(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def train_tokenizer(path, texts, vocab):
    tokenizer = Tokenizer(models.BPE(unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab, special_tokens=SPECIAL,
                                 initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.save(str(path))
    return tokenizer


def encode_texts(tokenizer, texts, max_len):
    encoded = tokenizer.encode_batch(texts)
    if any(len(e.ids) > max_len-2 for e in encoded):
        raise ValueError('An example exceeds max_len. Filter long examples or increase --max-len; no silent truncation.')
    return [[BOS]+e.ids+[EOS] for e in encoded]


def padded(rows, device='cpu'):
    result = torch.full((len(rows), max(map(len,rows))), PAD, dtype=torch.long)
    for i, row in enumerate(rows):
        result[i,:len(row)] = torch.tensor(row)
    return result.to(device)


def corrupt(ids, rate=0.3):
    eligible = ids.ge(5)
    # Independent token masking; span masking is a later ablation.
    mask = (torch.rand(ids.shape, device=ids.device) < rate) & eligible
    # Every nonempty sentence has at least one masked content token.
    for i in range(len(ids)):
        positions = eligible[i].nonzero().flatten()
        if len(positions) and not mask[i].any():
            mask[i,positions[torch.randint(len(positions),(1,),device=ids.device)]] = True
    return ids.masked_fill(mask, MASK)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
