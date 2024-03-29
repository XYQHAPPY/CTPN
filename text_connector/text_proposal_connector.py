# -*- coding: utf-8-*-
import numpy as np
from scipy.optimize import leastsq
from other import clip_boxes
from text_proposal_graph_builder import TextProposalGraphBuilder


def func(p, x):
    k, b = p
    return k * x + b


def error(p, x, y):
    return func(p, x) - y

class TextProposalConnector:
    """
        Connect text proposals into text lines
    """
    def __init__(self):
        self.graph_builder=TextProposalGraphBuilder()

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
    #leastaq
    def get_text_lines_leastaq(self, text_proposals, scores, im_size):
        tp_groups = self.group_text_proposals(text_proposals, scores, im_size)
        text_lines = np.zeros((len(tp_groups), 9), np.float32)
        for index, tp_indices in enumerate(tp_groups):
            text_line_boxes=text_proposals[list(tp_indices)]
            x0=np.min(text_line_boxes[:, 0])
            x1=np.max(text_line_boxes[:, 2])
            p0 = [x0,x1]
            Para1 = leastsq(error, p0,args=(np.array(text_line_boxes[:, 0]),np.array(text_line_boxes[:, 1])))
            Para2 = leastsq(error, p0, args=(np.array(text_line_boxes[:, 2]),np.array(text_line_boxes[:, 3])))
            k1, b1 = Para1[0]
            k2, b2 = Para2[0]
            y10 = k1 * x0 + b1
            y11 = k1 * x1 + b1
            y20 = k2 * x0 + b2
            y21 = k2 * x1 + b2
            score = scores[list(tp_indices)].sum() / float(len(tp_indices))
            text_lines[index, 0] = x0
            text_lines[index, 1] = y10
            text_lines[index, 2] = x1
            text_lines[index, 3] = y11
            text_lines[index, 4] = x0
            text_lines[index, 5] = y20
            text_lines[index, 6] = x1
            text_lines[index, 7] = y21
            text_lines[index, 8] = score
        return text_lines

    def get_text_lines(self, text_proposals, scores, im_size):
        # tp=text proposal
        tp_groups=self.group_text_proposals(text_proposals, scores, im_size)
        text_lines=np.zeros((len(tp_groups), 5), np.float32)

        for index, tp_indices in enumerate(tp_groups):
            text_line_boxes=text_proposals[list(tp_indices)]

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

        text_lines=clip_boxes(text_lines, im_size)

        return text_lines
