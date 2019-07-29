# -*- coding: utf-8 -*-
"""
@author: limingfan

"""

import model_comm_meta as meta

import model_detect_data
from model_detect_wrap import ModelDetect
import numpy as np
from text_connector.text_proposal_connector import TextProposalConnector
from text_connector.other import draw_boxes,normalize
from text_connector.cpu_nms import cpu_nms as nms
import os
import cv2
from PIL import Image
#
os.environ['CUDA_VISIBLE_DEVICES'] = '0' #使用 GPU 0
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1' # 使用 GPU 0，1
#


#
model = ModelDetect()
#


#
# train
model.train_and_valid()
#

#
# validate
model.validate(1000, False)
#
#

def predict():
    model.load_pb_for_prediction()
    sess = model.create_session_for_prediction()
    count = 0
    #
    list_images_valid = model_detect_data.getFilesInDirect(meta.dir_images_valid, meta.str_dot_img_ext)
    for img_file in list_images_valid:
        #
        # img_file = './data_test/images/bkgd_1_0_generated_0.png'
        #
        print(img_file)
        im = cv2.imread(img_file)
        #
        text_bbox,text_score = model.predict(sess, img_file, out_dir = './results_prediction')
        text_bbox,text_score = np.array(text_bbox),np.array(text_score)
        if len(text_score) == 0:
            print 'no text_bbox'
            continue
        #
        sorted_indices = np.argsort(text_score)[::-1]
        text_proposals = text_bbox[sorted_indices]
        text_score = text_score[sorted_indices]

        keep_inds = nms(np.column_stack((text_proposals, text_score)), meta.TEXT_PROPOSALS_NMS_THRESH)
        text_proposals, text_score = text_proposals[keep_inds], text_score[keep_inds]
        # print "result2:" + str(text_proposals)
        # return text_proposals

        text_score = normalize(text_score)
        textProposalConnector = TextProposalConnector()
        text_lines = textProposalConnector.get_text_lines(text_proposals, text_score, im.shape[:2])
        # text_lines = self.text_proposal_connector.get_text_lines_leastaq(text_proposals, scores, im.shape[:2])
        kanyixia = img_file.split('/')
        draw_boxes(im, text_lines, is_display=False, caption=kanyixia[len(kanyixia)-1], wait=False)
        # img = Image.open(img_file)
        # for itemimg in text_lines[:,0:4]:
        #     img1 = img.crop(itemimg)
        #     img1.save("/data/sata/share_sata/xyq/crnn/image/"+str(count)+".png","PNG")
        #     count += 1


#predict()