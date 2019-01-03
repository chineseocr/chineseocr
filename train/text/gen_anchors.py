"""
Reference: https://github.com/qqwweee/keras-yolo3.git
"""

from glob import glob
from PIL import Image
import numpy as np
from apphelper.image import get_box_spilt,read_voc_xml,resize_im


class YOLO_Kmeans:

    def __init__(self, cluster_number,root,scales = [416,512,608,608,608,768,960,1024],splitW=8):
        self.cluster_number = cluster_number
        self.filenames =glob(root)
        self.scales = scales
        self.splitW = splitW
        boxWH       = self.voc2boxes()
        
        res = self.kmeans(np.array(boxWH), k=cluster_number)
        
        self.anchors = self.gen_anchors(sorted(res,key=lambda x:x[1]))
        
                 
    def gen_anchors(self, boxWH):
        row = np.shape(boxWH)[0]
        tmp=[]
        for i in range(row):
            x_y = "%d,%d" % (boxWH[i][0], boxWH[i][1])
            tmp.append(x_y)
        return ', '.join(tmp)
    
    def voc2boxes(self):
        boxWH = []
        for filename in self.filenames:
            boxWH.extend(self.get_xml_box_wh(filename))
              
        
        return boxWH
    
    def get_xml_box_wh(self,filename):
        xmlP  = filename.replace('.jpg','.xml').replace('.png','.xml')
        boxes = read_voc_xml(xmlP)
        im    = Image.open(filename)
        scale = np.random.choice(self.scales,1)[0]
        w,h = resize_im(im.size[0],im.size[1], scale=scale, max_scale=2048)
        input_shape = (h,w)
        isRoate=False
        rorateDegree=0 
        newBoxes,newIm = get_box_spilt(boxes,im,sizeW=w,SizeH=h,splitW=self.splitW,isRoate=isRoate,rorateDegree=rorateDegree)
        box = []
        for bx in newBoxes:
            w = int(bx[2]-bx[0])
            h = int(bx[3]-bx[1])
            box.append([w,h])
        return box
    


    def iou(self, boxes, clusters):  # 1 box -> k clusters
        n = boxes.shape[0]
        k = self.cluster_number

        box_area = boxes[:, 0] * boxes[:, 1]
        box_area = box_area.repeat(k)
        box_area = np.reshape(box_area, (n, k))

        cluster_area = clusters[:, 0] * clusters[:, 1]
        cluster_area = np.tile(cluster_area, [1, n])
        cluster_area = np.reshape(cluster_area, (n, k))

        box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))
        cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))
        min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)

        box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
        cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))
        min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)
        inter_area = np.multiply(min_w_matrix, min_h_matrix)

        result = inter_area / (box_area + cluster_area - inter_area)
        return result

    def avg_iou(self, boxes, clusters):
        accuracy = np.mean([np.max(self.iou(boxes, clusters), axis=1)])
        return accuracy

    def kmeans(self, boxes, k, dist=np.median):
        box_number = boxes.shape[0]
        distances = np.empty((box_number, k))
        last_nearest = np.zeros((box_number,))
        np.random.seed()
        clusters = boxes[np.random.choice(
            box_number, k, replace=False)]  # init k clusters
        while True:

            distances = 1 - self.iou(boxes, clusters)

            current_nearest = np.argmin(distances, axis=1)
            if (last_nearest == current_nearest).all():
                break  # clusters won't change
            for cluster in range(k):
                clusters[cluster] = dist(  # update clusters
                    boxes[current_nearest == cluster], axis=0)

            last_nearest = current_nearest

        return clusters






