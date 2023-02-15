from dioptra.miners.base_miner import BaseMiner
from dioptra.lake.utils import _list_miner_metadata

def list_miners():
    """
    List all the miners

    """

    miner_list = []

    for metadata in _list_miner_metadata():
        my_miner = BaseMiner()
        my_miner.miner_id = metadata['_id']
        my_miner.miner_name = metadata['display_name']
        miner_list.append(my_miner)

    return miner_list
