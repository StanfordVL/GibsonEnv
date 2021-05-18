import os
import shutil
import tarfile
from pathlib import Path

import requests
from tqdm import tqdm

from gibson.assets.assets_manager import AssetsManager


def set_assets_path():
    assets_manager = AssetsManager()
    path = input('Insert the new assets path\n')
    assets_manager.set_assets_path(path).save_assets_information()
    print('Assets path correctly changed!')


def _download_archive(url, save_path, archive_name):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    archive_name_temp = archive_name + '.tmp'

    if not os.path.exists(os.path.join(save_path, archive_name)):
        total_size = int(requests.get(url, stream=True, allow_redirects=True).headers.get('content-length', 0))

        if os.path.exists(os.path.join(save_path, archive_name_temp)):
            downloaded_bytes = os.stat(os.path.join(save_path, archive_name_temp)).st_size
            resume_header = {'Range': 'bytes=%d-' % downloaded_bytes}
            mode = 'ab'
        else:
            downloaded_bytes = 0
            resume_header = {'Range': 'bytes=%d-' % downloaded_bytes}
            mode = 'wb'

        response = requests.get(url, stream=True,  headers=resume_header, allow_redirects=True)

        with open(os.path.join(save_path, archive_name_temp), mode=mode) as file, tqdm(
                desc='Downloading Gibson assets core',
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
                initial=downloaded_bytes
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)

        os.rename(os.path.join(save_path, archive_name_temp), os.path.join(save_path, archive_name))

    print('Archive downloaded!')


def _decompress_archive(path, archive_name):
    tar = tarfile.open(os.path.join(path, archive_name), mode='r:gz')
    bar = tqdm(tar.getmembers())
    bar.set_description('Extracting files')
    for member in bar:
        tar.extract(member, path=path)


def _copy_tree(src_dir, dst_dir):
    for src_file in os.listdir(src_dir):
        try:
            shutil.move(os.path.join(src_dir, src_file), dst_dir)
        except Exception:
            if os.path.isdir(os.path.join(dst_dir, src_file)):
                shutil.rmtree(os.path.join(dst_dir, src_file))
            else:
                os.remove(os.path.join(dst_dir, src_file))
            shutil.move(os.path.join(src_dir, src_file), dst_dir)


def download_assets_core():
    assets_manager = AssetsManager()
    assets_path = assets_manager.get_assets_path()

    path = '/tmp/gibson'
    archive_name = 'assets_core_v2.tar.gz'

    _download_archive(AssetsManager.CORE_ASSETS_URL, path, archive_name)

    _decompress_archive(path, archive_name)
    print('Copy assets to Gibson assets folder at ' + assets_path)

    _copy_tree(os.path.join(path, 'assets'), assets_path)
    print('Completed!')


def download_dataset():
    assets_manager = AssetsManager()
    assets_path = assets_manager.get_assets_path() + '/dataset'

    if not os.path.exists(assets_path):
        os.mkdir(assets_path)

    path = '/tmp/gibson'
    archive_name = 'dataset.tar.gz'

    _download_archive(AssetsManager.DATASET_URL, path, archive_name)

    print('Starting to extract files')
    _decompress_archive(path, archive_name)

    print('Copy assets to Gibson dataset folder at ' + assets_path)

    _copy_tree(os.path.join(path, 'dataset'), assets_path)
    print('Completed!')