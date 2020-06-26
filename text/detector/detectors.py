#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from text.detector.nms import nms,rotate_nms
from apphelper.image import get_boxes

from text.detector.text_proposal_connector import TextProposalConnector

def normalize(data):
    if data.shape[0]==0:
        return data
    max_=data.max()
    min_=data.min()
    return (data-min_)/(max_-min_) if max_-min_!=0 else data-min_

class TextDetector:
    """
        Detect text from an image
    """
    def __init__(self,MAX_HORIZONTAL_GAP=30,MIN_V_OVERLAPS=0.6,MIN_SIZE_SIM=0.6):
        """
        pass
        """
        self.text_proposal_connector=TextProposalConnector(MAX_HORIZONTAL_GAP,MIN_V_OVERLAPS,MIN_SIZE_SIM)
            
        
    def detect(self, text_proposals,scores,size,
               TEXT_PROPOSALS_MIN_SCORE=0.7,
               TEXT_PROPOSALS_NMS_THRESH=0.3,
               TEXT_LINE_NMS_THRESH = 0.3,
               LINE_MIN_SCORE=0.8
               ):
        """
        Detecting texts from an image
        :return: the bounding boxes of the detected texts
        @@param:TEXT_PROPOSALS_MIN_SCORE:TEXT_PROPOSALS_MIN_SCORE=0.7##过滤字符box阀值
        @@param:TEXT_PROPOSALS_NMS_THRESH:TEXT_PROPOSALS_NMS_THRESH=0.3##nms过滤重复字符box
        @@param:TEXT_LINE_NMS_THRESH:TEXT_LINE_NMS_THRESH=0.3##nms过滤行文本重复过滤阀值
        @@param:LINE_MIN_SCORE:##行文本置信度
        
        """
        #text_proposals, scores=self.text_proposal_detector.detect(im, cfg.MEAN)
        keep_inds=np.where(scores>TEXT_PROPOSALS_MIN_SCORE)[0]### 
        text_proposals, scores=text_proposals[keep_inds], scores[keep_inds]
        sorted_indices=np.argsort(scores.ravel())[::-1]
        text_proposals, scores=text_proposals[sorted_indices], scores[sorted_indices]

        # nms for text proposals
        if len(text_proposals)>0:
            text_proposals, scores = nms(text_proposals,scores,TEXT_PROPOSALS_MIN_SCORE,TEXT_PROPOSALS_NMS_THRESH)
            scores=normalize(scores)            
            text_lines,scores = self.text_proposal_connector.get_text_lines(text_proposals, scores, size)##合并文本行
            text_lines = get_boxes(text_lines)
            text_lines, scores = rotate_nms(text_lines,scores,LINE_MIN_SCORE,TEXT_LINE_NMS_THRESH)
            
            return text_lines,scores
        else:
            return [],[]

