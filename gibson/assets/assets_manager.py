import os.path

import yaml


class AssetsManager:
    CONFIG_FILE_NAME = 'assets_config.yaml'
    KEY_ASSET_PATH = 'assets_path'

    def __init__(self):
        self._config_file_path = os.path.join(os.path.dirname(__file__), AssetsManager.CONFIG_FILE_NAME)

        self._assets_information = {AssetsManager.KEY_ASSET_PATH: ''}

        # Load assets information
        if os.path.exists(self._config_file_path):
            with open(self._config_file_path, mode='r') as f:
                self._assets_information = yaml.load(f, Loader=yaml.FullLoader)

    def set_assets_path(self, path):
        """
        Sets the assets data path which specifies the assets folder. This method also creates the main asset folder it does not exists
        :param path: the assets data path
        :type path: str
        :return: the AssetsManager instance
        """
        if not os.path.exists(os.path.dirname(path)):
            raise FileExistsError('The path where save the assets doesn\'t exists!!')

        if not os.path.exists(path):
            os.mkdir(path)

        dataset_path = os.path.exists(os.path.join(path, 'dataset'))
        if not dataset_path:
            os.mkdir(path)

        self._assets_information[AssetsManager.KEY_ASSET_PATH] = path

        return self

    def get_assets_path(self):
        """
        Returns the assets path
        :return:
        """
        return self._assets_information[AssetsManager.KEY_ASSET_PATH]

    def save_assets_information(self):
        """
        Saves the assets configration parameter to disk
        :return: the assets manager instance
        :rtype: AssetsManager
        """
        with open(self._config_file_path, mode='w') as f:
            yaml.dump(self._assets_information, f)

        return self
