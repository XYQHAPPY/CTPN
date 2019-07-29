# -*- coding: utf-8 -*-


# data for train
dir_data_train = './data_generated'
#dir_data_train = '/data/sata/share_sata/xyq/data/phrase_detection/train'
#
dir_images_train = dir_data_train + '/images'
dir_contents_train = dir_data_train + '/contents'

# data for validation
dir_data_valid = './data_test'
#dir_data_valid = '/data/sata/share_sata/xyq/data/phrase_detection/test'
#
dir_images_valid = dir_data_valid + '/images'
#dir_images_valid = '/data/sata/share_sata/xyq/data/phrase_source/军'
dir_contents_valid = dir_data_valid + '/contents'
#
dir_results_valid = dir_data_valid + '/results'
#
str_dot_img_ext = '.png'
#

#
model_detect_dir = './model_detect'
model_detect_name = 'model_detect'
model_detect_pb_file = model_detect_name + '.pb'
#
anchor_heights = [20, 22,24,26,28,30,32,34, 36, 38,40,42, 44]

#
threshold = 0.5  #
#

#
model_recog_dir = './model_recog'
model_recog_name = 'model_recog'
model_recog_pb_file = model_recog_name + '.pb'
#
height_norm = 24  #
#
MAX_HORIZONTAL_GAP=50
MIN_V_OVERLAPS=0.7
MIN_SIZE_SIM=0.7
TEXT_PROPOSALS_NMS_THRESH=0.3


