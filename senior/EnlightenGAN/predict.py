import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from pdb import set_trace as st
from util import html
import shutil
opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)
# create website
web_dir = opt.results_dir
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
# test
print(len(dataset))
for i, data in enumerate(dataset):
    model.set_input(data)
    visuals = model.predict()
    img_path = model.get_image_paths()
    print('process image... %s' % img_path)
    visualizer.save_images(webpage, visuals, img_path)

webpage.save()


for file in os.listdir(opt.results_dir+'/images'):
    file_path = os.path.join(opt.results_dir+'/images', file)
    if file.find("_fake_B.png") != -1:
        shutil.copy(file_path, opt.results_dir+'/'+file.split('_fake_B')[0]+'.png')

shutil.rmtree(opt.results_dir+'/images')
os.unlink(opt.results_dir+'/index.html')




    

