import os

from dataset import DataLoaderTrain_stage0, DataLoaderTrain_stage1, DataLoaderTrain, DataLoaderVal, DataLoaderTest, DataLoaderTestSR
# def get_training_data(rgb_dir, img_options):
#     assert os.path.exists(rgb_dir)
#     return DataLoaderTrain(rgb_dir, img_options, None)
def get_training_data(rgb_dir, img_options, stage=0):
    assert os.path.exists(rgb_dir)
    if stage == 1:
        return DataLoaderTrain_stage1(rgb_dir, img_options, None)
    else:
        return DataLoaderTrain_stage0(rgb_dir, img_options, None)

def get_validation_data(rgb_dir):
    assert os.path.exists(rgb_dir)
    return DataLoaderVal(rgb_dir, None)


def get_test_data(rgb_dir):
    assert os.path.exists(rgb_dir)
    return DataLoaderTest(rgb_dir, None)


def get_test_data_SR(rgb_dir):
    assert os.path.exists(rgb_dir)
    return DataLoaderTestSR(rgb_dir, None)