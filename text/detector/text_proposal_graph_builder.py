import numpy as np
class Graph:
    def __init__(self, graph):
        self.graph=graph

    def sub_graphs_connected(self):
        sub_graphs=[]
        for index in range(self.graph.shape[0]):
            if not self.graph[:, index].any() and self.graph[index, :].any():
                v=index
                sub_graphs.append([v])
                while self.graph[v, :].any():
                    v=np.where(self.graph[v, :])[0][0]
                    sub_graphs[-1].append(v)
        return sub_graphs
    

class TextProposalGraphBuilder:
    """
        Build Text proposals into a graph.
    """
    def __init__(self,MAX_HORIZONTAL_GAP=30,MIN_V_OVERLAPS=0.6,MIN_SIZE_SIM=0.6):
        """
        @@param:MAX_HORIZONTAL_GAP:文本行间隔最大值
        @@param:MIN_V_OVERLAPS
        @@param:MIN_SIZE_SIM
        MIN_V_OVERLAPS=0.6
        MIN_SIZE_SIM=0.6
        """
        self.MAX_HORIZONTAL_GAP = MAX_HORIZONTAL_GAP
        self.MIN_V_OVERLAPS = MIN_V_OVERLAPS
        self.MIN_SIZE_SIM = MIN_SIZE_SIM
        
    def get_successions(self, index):
            box=self.text_proposals[index]
            results=[]
            for left in range(int(box[0])+1, min(int(box[0])+self.MAX_HORIZONTAL_GAP+1, self.im_size[1])):
                adj_box_indices=self.boxes_table[left]
                for adj_box_index in adj_box_indices:
                    if self.meet_v_iou(adj_box_index, index):
                        results.append(adj_box_index)
                if len(results)!=0:
                    return results
            return results

    def get_precursors(self, index):
        box=self.text_proposals[index]
        results=[]
        for left in range(int(box[0])-1, max(int(box[0]-self.MAX_HORIZONTAL_GAP), 0)-1, -1):
            adj_box_indices=self.boxes_table[left]
            for adj_box_index in adj_box_indices:
                if self.meet_v_iou(adj_box_index, index):
                    results.append(adj_box_index)
            if len(results)!=0:
                return results
        return results

    def is_succession_node(self, index, succession_index):
        precursors=self.get_precursors(succession_index)
        if self.scores[index]>=np.max(self.scores[precursors]):
            return True
        return False

    def meet_v_iou(self, index1, index2):
        def overlaps_v(index1, index2):
            h1=self.heights[index1]
            h2=self.heights[index2]
            y0=max(self.text_proposals[index2][1], self.text_proposals[index1][1])
            y1=min(self.text_proposals[index2][3], self.text_proposals[index1][3])
            return max(0, y1-y0+1)/min(h1, h2)

        def size_similarity(index1, index2):
            h1=self.heights[index1]
            h2=self.heights[index2]
            return min(h1, h2)/max(h1, h2)

        return overlaps_v(index1, index2)>=self.MIN_V_OVERLAPS and \
               size_similarity(index1, index2)>=self.MIN_SIZE_SIM

    def build_graph(self, text_proposals, scores, im_size):
        self.text_proposals=text_proposals
        self.scores=scores
        self.im_size=im_size
        self.heights=text_proposals[:, 3]-text_proposals[:, 1]+1
        boxes_table=[[] for _ in range(self.im_size[1])]
        for index, box in enumerate(text_proposals):
            #print(int(box[0]),len(boxes_table))
            boxes_table[int(box[0])].append(index)
        self.boxes_table=boxes_table

        graph=np.zeros((text_proposals.shape[0], text_proposals.shape[0]), np.bool)

        for index, box in enumerate(text_proposals):
            successions=self.get_successions(index)
            if len(successions)==0:
                continue
            succession_index=successions[np.argmax(scores[successions])]
            if self.is_succession_node(index, succession_index):
                # NOTE: a box can have multiple successions(precursors) if multiple successions(precursors)
                # have equal scores.
                graph[index, succession_index]=True
        return Graph(graph)
