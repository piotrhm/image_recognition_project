from typing import Dict, Any

import os
import re
import torch
import requests
import torch.nn as nn

from tqdm import tqdm


def download_checkpoint(checkpoint_url: str, checkpoint_target_path: str) -> str:
    if 'drive.google.com' in checkpoint_url:
        session = requests.session()
        file_id = checkpoint_url.split('/')[5]
        url = f'https://drive.google.com/uc?export=download&id={file_id}'
        with session.get(url, allow_redirects=True) as r:
            assert r.status_code == 200
        confirmation = session.cookies[session.cookies.keys()[0]]
        with session.get(url + f'&confirm={confirmation}', stream=True) as r:
            r.raise_for_status()
            print('Downloading the checkpoint...', end=' ')
            with open(checkpoint_target_path, 'wb') as f:
                for chunk in tqdm(r.iter_content(chunk_size=8192)):
                    f.write(chunk)
    else:
        raise NotImplementedError(f'No handler for url: {checkpoint_url}')
    return checkpoint_target_path


def load_model(model_str: str, checkpoint_path: str, device: str = "cuda", model_kwargs: Dict[str, Any] = {}) -> nn.Module:
    if model_str.lower() == 'protopnet':
        os.system('git clone https://github.com/cfchen-duke/ProtoPNet.git')
        with open('ProtoPNet/model.py', 'r') as f:
            text = f.read()
        with open('ProtoPNet/model.py', 'w') as f:
            f.write(re.sub(r'^from (\w+_features|receptive_field)', r'from .\1', text, flags=re.MULTILINE))
        from ProtoPNet.model import construct_PPNet
        args = dict({'base_architecture': 'resnet34', 'add_on_layers_type': 'regular'}, **model_kwargs)
        model = construct_PPNet(**args)
    else:
        raise NotImplementedError(f'No model for string: {model_str}')
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path))
    return model
