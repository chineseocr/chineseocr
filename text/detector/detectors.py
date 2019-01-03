#coding:utf-8
import numpy as np
from config import GPUID,GPU
from text.detector.utils.cython_nms import nms as cython_nms
from text.detector.text_proposal_connector import TextProposalConnector

##优先加载编译对GPU编译的gpu_nms 如果不想调用GPU，在程序启动执行os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if GPU:
    try:
        from detector.utils.gpu_nms import gpu_nms
    except:
        gpu_nms =cython_nms

def nms(dets, thresh):
    if dets.shape[0] == 0:
        return []
    
    try:
        if GPU and GPUID is not None:
           return gpu_nms(dets, thresh, device_id=GPUID)
    except:
            return cython_nms(dets, thresh)

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
               MIN_RATIO=1.0,
               LINE_MIN_SCORE=0.8,
               TEXT_PROPOSALS_WIDTH=5,
               MIN_NUM_PROPOSALS=1
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
            
            text_lines=self.text_proposal_connector.get_text_lines(text_proposals, scores, size)##合并文本行
            return text_lines
        else:
            return []

