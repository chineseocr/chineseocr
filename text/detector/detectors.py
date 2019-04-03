#coding:utf-8
import numpy as np
from config import GPUID,GPU,nmsFlag
from text.detector.utils.python_nms import nms as python_nms ##python版本nms

from text.detector.text_proposal_connector import TextProposalConnector


##优先加载编译对GPU编译的gpu_nms

if nmsFlag=='gpu' and GPU and GPUID is not None:
    try:
        from text.detector.utils.gpu_nms import gpu_nms
    except:
            gpu_nms = None
    cython_nms = None

elif nmsFlag=='python':
     gpu_nms ==None
     cython_nms = None

elif nmsFlag=='cython':
    try:
        from text.detector.utils.cython_nms  import nms as  cython_nms
    except:
         cython_nms = None
    gpu_nms =None
else:
    gpu_nms =None
    cython_nms = None

print("Nms engine gpu_nms:",gpu_nms,",cython_nms:",cython_nms,",python_nms:",python_nms)


def nms(dets, thresh):
    if dets.shape[0] == 0:
        return []
    
    if  gpu_nms is not None and GPUID is not None:
        return gpu_nms(dets, thresh, device_id=GPUID)

    elif  cython_nms is not None:
          return cython_nms(dets, thresh)
    else:
          return python_nms(dets, thresh, method='Union')


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
        
    def detect_region(self, text_proposals,scores,size,
               TEXT_PROPOSALS_MIN_SCORE=0.7,
               TEXT_PROPOSALS_NMS_THRESH=0.3,
               TEXT_LINE_NMS_THRESH = 0.3,):
        """
        Detecting texts from an image
        :return: the bounding boxes of the detected texts
        @@param:TEXT_PROPOSALS_MIN_SCORE:TEXT_PROPOSALS_MIN_SCORE=0.7##过滤字符box阀值
        @@param:TEXT_PROPOSALS_NMS_THRESH:TEXT_PROPOSALS_NMS_THRESH=0.3##nms过滤重复字符box
        @@param:TEXT_LINE_NMS_THRESH:TEXT_LINE_NMS_THRESH=0.3##nms过滤行文本重复过滤阀值
        @@param:MIN_RATIO:MIN_RATIO=1.0#0.01 ##widths/heights宽度与高度比例
        @@param:LINE_MIN_SCORE:##行文本置信度
        @@param:TEXT_PROPOSALS_WIDTH##每个字符的默认最小宽度
        @@param:MIN_NUM_PROPOSALS,MIN_NUM_PROPOSALS=1##最小字符数
        """
        #text_proposals, scores=self.text_proposal_detector.detect(im, cfg.MEAN)
        keep_inds=np.where(scores>TEXT_PROPOSALS_MIN_SCORE)[0]###
        text_proposals, scores=text_proposals[keep_inds], scores[keep_inds]

        sorted_indices=np.argsort(scores.ravel())[::-1]
        text_proposals, scores=text_proposals[sorted_indices], scores[sorted_indices]

        # nms for text proposals
        keep_inds=nms(np.hstack((text_proposals, scores)), TEXT_PROPOSALS_NMS_THRESH,GPU_ID=self.GPU_ID)##nms 过滤重复的box 
        text_proposals, scores=text_proposals[keep_inds], scores[keep_inds]
        
        groups_boxes,groups_scores = self.text_proposal_connector.get_text_region(text_proposals, scores, size)
        return groups_boxes,groups_scores
    
    
        
    def detect(self, text_proposals,scores,size,
               TEXT_PROPOSALS_MIN_SCORE=0.7,
               TEXT_PROPOSALS_NMS_THRESH=0.3,
               TEXT_LINE_NMS_THRESH = 0.3,
               
               ):
        """
        Detecting texts from an image
        :return: the bounding boxes of the detected texts
        @@param:TEXT_PROPOSALS_MIN_SCORE:TEXT_PROPOSALS_MIN_SCORE=0.7##过滤字符box阀值
        @@param:TEXT_PROPOSALS_NMS_THRESH:TEXT_PROPOSALS_NMS_THRESH=0.3##nms过滤重复字符box
        @@param:TEXT_LINE_NMS_THRESH:TEXT_LINE_NMS_THRESH=0.3##nms过滤行文本重复过滤阀值
        @@param:MIN_RATIO:MIN_RATIO=1.0#0.01 ##widths/heights宽度与高度比例
        @@param:LINE_MIN_SCORE:##行文本置信度
        @@param:TEXT_PROPOSALS_WIDTH##每个字符的默认最小宽度
        @@param:MIN_NUM_PROPOSALS,MIN_NUM_PROPOSALS=1##最小字符数
        
        """
        #text_proposals, scores=self.text_proposal_detector.detect(im, cfg.MEAN)
        keep_inds=np.where(scores>TEXT_PROPOSALS_MIN_SCORE)[0]###
        
        text_proposals, scores=text_proposals[keep_inds], scores[keep_inds]

        sorted_indices=np.argsort(scores.ravel())[::-1]
        text_proposals, scores=text_proposals[sorted_indices], scores[sorted_indices]

        # nms for text proposals
        if len(text_proposals)>0:
            keep_inds=nms(np.hstack((text_proposals, scores)), TEXT_PROPOSALS_NMS_THRESH)##nms 过滤重复的box 
            text_proposals, scores=text_proposals[keep_inds], scores[keep_inds]

            scores=normalize(scores)
            
            text_lines = self.text_proposal_connector.get_text_lines(text_proposals, scores, size)##合并文本行
            keep_inds  = nms(text_lines, TEXT_LINE_NMS_THRESH)##nms
            text_lines = text_lines[keep_inds]
            return text_lines
        else:
            return []

