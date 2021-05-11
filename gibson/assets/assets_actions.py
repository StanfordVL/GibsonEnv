from gibson.assets.assets_manager import AssetsManager


def set_assets_path():
    assets_manager = AssetsManager()
    print('Current assets path = ' + assets_manager.get_assets_path())
    path = input('Insert the new assets path\n')
    assets_manager.set_assets_path(path).save_assets_information()
    print('Assets path correctly changed!')