import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from latentlab.data import make_data, read_pairs
from latentlab.experiment import train, align, evaluate, export, fit_map, apply_map, load_model

class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_coordinate_recovery(self):
        torch.manual_seed(10)
        x=torch.randn(80,2,8)
        q,_=torch.linalg.qr(torch.randn(8,8))
        y=x@q+torch.randn(8)
        for kind in ['orthogonal','linear']:
            bridge=fit_map(x[:60],y[:60],kind,ridge=1e-8)
            self.assertLess((apply_map(x[60:],bridge)-y[60:]).abs().max().item(),1e-4)

    def test_data_separation(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'data'
            make_data(out,train=100,val=10,align=10,test=10,ood=10)
            training=set((out/'train.en.txt').read_text().splitlines())
            seen=set()
            for split in ['val','align','test','ood']:
                for row in read_pairs(out/f'{split}.jsonl'):
                    event=tuple(row['event'])
                    self.assertNotIn(event,seen)
                    self.assertNotIn(row['en'],training)
                    seen.add(event)
                    self.assertEqual((event[0],event[2]) in {(0,5),(5,3)},split=='ood')

    def test_train_resume_swap_align_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); data=root/'data'
            make_data(data,train=32,val=8,align=8,test=8,ood=4)
            base=dict(data=str(data),device='cpu',tokenizer=None,epochs=1,batch_size=8,
                      seed=1,vocab_size=512,width=32,heads=4,layers=1,slots=2,
                      max_len=96,dropout=0.0,lr=0.001,mask_rate=0.3,resume=False,fp32=True)
            en=SimpleNamespace(**base,lang='en',out=str(root/'en'))
            es=SimpleNamespace(**base,lang='es',out=str(root/'es'))
            train(en); train(es)
            en.resume=True; en.epochs=2
            train(en)
            model,tok,checkpoint=load_model(root/'en'/'last.pt','cpu')
            self.assertEqual(checkpoint['epoch'],2)
            en_path=str(root/'en'/'best.pt'); es_path=str(root/'es'/'best.pt')
            mapping=str(root/'bridge.pt')
            align(SimpleNamespace(source=en_path,target=es_path,pairs=str(data/'align.jsonl'),
                                  out=mapping,n=8,seed=42,kind='orthogonal',ridge=0.01,
                                  shuffle_pairs=False,batch_size=4,device='cpu'))
            for name,target,bridge,shuffle in [('reconstruct',en_path,None,False),
                                               ('direct',es_path,None,False),
                                               ('aligned',es_path,mapping,False),
                                               ('shuffled',en_path,None,True)]:
                dest=root/f'{name}.json'
                evaluate(SimpleNamespace(source=en_path,target=target,pairs=str(data/'test.jsonl'),
                    out=str(dest),bridge=bridge,shuffle_latents=shuffle,limit=4,batch_size=2,device='cpu'))
                result=json.loads(dest.read_text())
                self.assertEqual(result['n'],4)
                self.assertTrue(0<=result['chrf']<=100)
                self.assertEqual(len(read_pairs(dest.with_suffix('.predictions.jsonl'))),4)
            export(SimpleNamespace(checkpoint=en_path,out=str(root/'export')))
            self.assertTrue((root/'export'/'encoder.pt').exists())
            # The decoder must remain sensitive to latent conditioning.
            z=torch.randn(2,2,32); prefix=torch.tensor([[1,6],[1,6]])
            self.assertGreater((model.decode(prefix,z)-model.decode(prefix,z.flip(0))).abs().max().item(),0.01)

if __name__=='__main__':
    unittest.main()
