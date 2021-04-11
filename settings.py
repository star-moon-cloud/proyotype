base_architecture = 'vgg19'
img_size = 224
prototype_shape = (60, 128, 1, 1)
num_classes = 10
prototype_activation_function = 'log'
add_on_layers_type = 'regular'
inst_shape = (20,128,1,1)
clst_shape = (40,128,1,1)
experiment_run = '实验三十1'

data_path = './datasets/cub200_cropped/'
train_dir = data_path + 'train_cropped_augmented1/'
test_dir = data_path + 'test_cropped1/'
train_push_dir = data_path + 'train_cropped1/'
train_inst_dir = data_path + 'instruction_augmented1/'
train_batch_size = 40
test_batch_size = 100
train_push_batch_size = 75

joint_optimizer_lrs = {'features': 1e-4,
                       'add_on_layers': 3e-3,
                       'prototype_vectors': 3e-3}
joint_lr_step_size = 5

warm_optimizer_lrs = {'add_on_layers': 3e-3,
                      'prototype_vectors': 3e-3}

last_layer_optimizer_lr = 1e-4
'''
                       
joint_optimizer_lrs = {'features': 0.8e-4, 
                       'add_on_layers': 2.4e-3,
                       'prototype_vectors':2.4e-3}
joint_lr_step_size = 5

warm_optimizer_lrs = {'add_on_layers': 2.4e-3,
                      'prototype_vectors': 2.4e-3}

last_layer_optimizer_lr = 0.8e-4'''
coefs = {
    'crs_ent': 1,
    'clst': 0.8,
    'clst1':0.8,
    'sep': -0.08,
    'inst': 1,
    'l1': 1e-4,
    'sim':0
}

num_train_epochs = 100
num_warm_epochs = 3

push_start = 10
push_epochs = [i for i in range(num_train_epochs) if i % 10 == 0]
