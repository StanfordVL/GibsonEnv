import os

import requests
from tqdm import tqdm

from gibson.assets.assets_manager import AssetsManager


def set_assets_path():
    assets_manager = AssetsManager()
    path = input('Insert the new assets path\n')
    assets_manager.set_assets_path(path).save_assets_information()
    print('Assets path correctly changed!')


def download_assets_core():
    assets_manager = AssetsManager()
    assets_path = assets_manager.get_assets_path()

    assets_core_path = '/tmp/assets_core_v2.tar.gz'
    if os.path.exists(assets_path):
        total = int(requests.get(AssetsManager.CORE_ASSETS_URL, stream=True, verify=False,  allow_redirects=True).headers.get('content-length', 0))

        resume_header = {'Range': 'bytes=%d-' % os.stat(assets_core_path).st_size}
        print(resume_header)
        resp = requests.get(AssetsManager.CORE_ASSETS_URL, stream=True,  headers=resume_header, allow_redirects=True)
        with open('/tmp/assets_core_v2.tar.gz', 'ab') as file, tqdm(
                desc='Downloading Gibson assets core',
                total=total,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
                initial=os.stat(assets_core_path).st_size
        ) as bar:
            for data in resp.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
