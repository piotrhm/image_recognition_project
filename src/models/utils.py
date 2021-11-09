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


def load_model(model: nn.Module, checkpoint_path: str, device: str = "cuda") -> nn.Module:
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path))
    return model
