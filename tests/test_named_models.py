from dataclasses import asdict
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import torch
from latentlab.app import merge_named, MergedModel, generate_named, train_named
from latentlab.data import train_tokenizer, padded, encode_texts, make_data
from latentlab.model import Autoencoder, Config
from latentlab.experiment import atomic_save

class NamedModelTests(unittest.TestCase):
    def test_merge_preserves_selected_computation_and_is_standalone(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            models=[]
            for name,lang in [('one','en'),('two','es')]:
                folder=root/name;folder.mkdir()
                tokenizer=train_tokenizer(folder/'tokenizer.json',['The red dog follows the blue cat.'],400)
                model=Autoencoder(Config(tokenizer.get_vocab_size(),width=32,layers=1,slots=2,max_len=64,dropout=0)).eval()
                atomic_save({'config':asdict(model.config),'model':model.state_dict(),
                             'tokenizer':tokenizer.to_str(),'lang':lang,'epoch':1},folder/'best.pt')
                models.append((model,tokenizer))
            path=merge_named('one','two','combined',root)
            merged=MergedModel(path,'cpu')
            text='The red dog follows the blue cat.'
            ids=padded(encode_texts(models[0][1],[text],64))
            with torch.no_grad():
                original_z=models[0][0].encode(ids)
                torch.testing.assert_close(merged.encoder.encode(ids),original_z)
                prefix=torch.tensor([[1,5,6]])
                torch.testing.assert_close(merged.decoder.decode(prefix,original_z),models[1][0].decode(prefix,original_z))
            first=merged.generate(text)
            shutil.rmtree(root/'one');shutil.rmtree(root/'two')
            self.assertEqual(generate_named('combined',text,root,'cpu'),first)
            with self.assertRaises(ValueError): merge_named('one','two','combined',root)

    def test_named_training_and_resume(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'data'
            make_data(data,train=16,val=4,test=4,ood=4)
            with patch.dict('latentlab.app.DEFAULTS',dict(width=32,layers=1,slots=2,max_len=96,vocab_size=512,dropout=0)):
                train_named('english','my_english',root/'models',data,epochs=1,batch_size=8,seed=3,device='cpu')
            train_named('english','my_english',root/'models',resume=True,epochs=2,device='cpu')
            state=torch.load(root/'models/my_english/last.pt',weights_only=True)
            self.assertEqual(state['epoch'],2)
            self.assertEqual(state['config']['width'],32)
            with self.assertRaises(ValueError):
                train_named('spanish','my_english',root/'models',resume=True,epochs=3)
