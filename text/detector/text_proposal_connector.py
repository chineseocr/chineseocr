#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from text.detector.text_proposal_graph_builder import TextProposalGraphBuilder

class TextProposalConnector:
    """
        Connect text proposals into text lines
    """
    def __init__(self,MAX_HORIZONTAL_GAP=30,MIN_V_OVERLAPS=0.6,MIN_SIZE_SIM=0.6):
        self.graph_builder=TextProposalGraphBuilder(MAX_HORIZONTAL_GAP,MIN_V_OVERLAPS,MIN_SIZE_SIM)

    def group_text_proposals(self, text_proposals, scores, im_size):
        graph=self.graph_builder.build_graph(text_proposals, scores, im_size)
        return graph.sub_graphs_connected()

    def fit_y(self, X, Y, x1, x2):
        len(X)!=0
        # if X only include one point, the function will get line y=Y[0]
        if np.sum(X==X[0])==len(X):
            return Y[0], Y[0]
        p=np.poly1d(np.polyfit(X, Y, 1))
        return p(x1), p(x2)

    def get_text_lines(self, text_proposals, scores, im_size):
        """
        text_proposals:boxes
        
        """
        # tp=text proposal
        tp_groups=self.group_text_proposals(text_proposals, scores, im_size)##find the text line 
        
        text_lines=np.zeros((len(tp_groups), 8), np.float32)
        newscores =np.zeros((len(tp_groups), ), np.float32)
        for index, tp_indices in enumerate(tp_groups):
            text_line_boxes=text_proposals[list(tp_indices)]
            #num = np.size(text_line_boxes)##find 
            X = (text_line_boxes[:,0] + text_line_boxes[:,2]) / 2
            Y = (text_line_boxes[:,1] + text_line_boxes[:,3]) / 2
            
            z1 = np.polyfit(X,Y,1)
           # p1 = np.poly1d(z1)


            x0=np.min(text_line_boxes[:, 0])
            x1=np.max(text_line_boxes[:, 2])

            offset=(text_line_boxes[0, 2]-text_line_boxes[0, 0])*0.5

            lt_y, rt_y=self.fit_y(text_line_boxes[:, 0], text_line_boxes[:, 1], x0+offset, x1-offset)
            lb_y, rb_y=self.fit_y(text_line_boxes[:, 0], text_line_boxes[:, 3], x0+offset, x1-offset)

            # the score of a text line is the average score of the scores
            # of all text proposals contained in the text line
            score=scores[list(tp_indices)].sum()/float(len(tp_indices))

            text_lines[index, 0]=x0
            text_lines[index, 1]=min(lt_y, rt_y)
            text_lines[index, 2]=x1
            text_lines[index, 3]=max(lb_y, rb_y)
            text_lines[index, 4]=score
            text_lines[index, 5]=z1[0]
            text_lines[index, 6]=z1[1]
            height = np.mean( (text_line_boxes[:,3]-text_line_boxes[:,1]) )
            text_lines[index, 7]= height + 2.5
            newscores[index] = score



        return text_lines,newscores
