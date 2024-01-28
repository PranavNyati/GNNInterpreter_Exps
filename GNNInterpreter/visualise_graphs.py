from re import S
import numpy as np
import torch
import math
import tqdm
import sys
import matplotlib.pyplot as plt
import networkx as nx
import os
np.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(threshold=sys.maxsize)

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../.venv/lib/python3.8/site-packages/'))

# import dgl

save_path = './Exp_graph_plots_mutag_full/'




NODE_CLS = {
    0: 'C',
    1: 'N',
    2: 'O',
    3: 'F',
    4: 'I',
    5: 'Cl',
    6: 'Br',
}

REV_NODE_CLS = {
    'C': 0,
    'N': 1,
    'O': 2,
    'F': 3,
    'I': 4,
    'Cl': 5,
    'Br': 6,
}

NODE_COLOR = {
    0: 'orange',
    1: 'green',
    2: 'yellow',
    3: 'blue',
    4: 'magenta',
    5: 'yellowgreen',
    6: 'brown',
}

COLOR_TO_NODE_MAP = {
    'orange': 'C',
    'green': 'N',
    'red': 'O',
    'blue': 'F',
    'magenta': 'I',
    'yellowgreen': 'Cl',
    'brown': 'Br'
}

GRAPH_CLS = {
    0: 'nonmutagen',
    1: 'mutagen',
}


EDGE_CLS = {
    1: 'single',
    2: 'double',
    3: 'triple'
}

edge_weight_map = {
    1: 3,
    2: 6,
    3: 9
}

# edge_exp_map = {
#     0.0: 'single_exp',
#     1.0: 'double_exp',
#     2.0: 'triple_exp'
# }

# creating a color map for the edges
edge_color_dict = {
    1: 'tera cotta',
    2: 'dark slate gray',
    3: 'golden',
}



def mutag_dgl_to_networkx(dgl_G):
    component_dict = {0: 'C', 1: 'O', 2: 'Cl', 3: 'H', 4: 'N', 5: 'F', 6: 'Br', 7: 'S',
                      8: 'P', 9: 'I', 10: 'Na', 11: 'K', 12: 'Li', 13: 'Ca'}
    nodes = dgl_G.nodes().numpy()
    edges = np.array(list(zip(dgl_G.edges()[0], dgl_G.edges()[1])))
    node_labels = dgl_G.ndata['feat'].numpy()
    edge_weights = dgl_G.edata['weight'].numpy()
    edge_labels = dgl_G.edata['label'].numpy()
    edges = edges[np.where(edge_weights > 0)]
    edge_labels = edge_labels[np.where(edge_weights > 0)]
    nx_G = nx.Graph()
    nx_G.add_nodes_from(nodes)
    # add edge with label
    for eid in range(len(edges)):
        nx_G.add_edge(edges[eid][0], edges[eid][1], gt=edge_labels[eid])
    for node in nx_G.nodes(data=True):
        node[1]['label'] = component_dict[np.where(node_labels[node[0]] == 1.0)[0][0]]
        # print(node[0] , ": ", node[1]['label'])
    return nx_G


# def get_mutag_color_dict():
#     mutage_color_dict = {'C': 'tab:orange', 'O': 'tab:gray', 'Cl': 'cyan', 'H': 'tab:blue', 'N': 'blue',
#                        'F': 'green', 'Br': 'y', 'S': 'm', 'P': 'red', 'I': 'tab:green', 'Na': 'tab: purple',
#                        'K': 'tab:brown', 'Li': 'tab:pink', 'Ca': 'tab:olive'}
#     return mutage_color_dict


def get_graph_node_color_encodings(nx_graph):
    node_labels = {}
    node_color_coding = {}

    # find the node_label each from the one-hot encoding
    for i in range(nx_graph.number_of_nodes()):
        node_label = nx_graph.nodes[i]['label']

        node_label_int = REV_NODE_CLS[node_label]
        node_color_coding[str(i)]=dict(color=COLOR_TO_NODE_MAP[node_label])
    
        

    # print(node_labels)   
    print("Node color coding: ", node_color_coding)

    
    return node_color_coding

def create_nx_graph(nx_graph, node_color_coding, explanation_edges, tempfile):
    
    graph_viz = nx.Graph()
    graph_viz.add_nodes_from(n for n in node_color_coding.items())
    
    print("GRAPH VISUALISATION: \n")
    print(graph_viz.number_of_nodes())
    # print(graph_viz.nodes.data())
    # print("\n")
    
    
    edges = []
    
    if explanation_edges is not None:
        for edge in nx_graph.edges.data():
            # print(edge)
            if edge not in explanation_edges:
                print("Edge not in explanation edges: ", edge)
                if ( edge[0], edge[1], EDGE_CLS[edge[2]['label']] ) not in edges :
                    graph_viz.add_edge(edge[0], edge[1], color='blue', weight=edge_weight_map[edge[2]['label']])
            else:
                if (edge[0], edge[1], EDGE_CLS[edge[2]['label']]) not in edges:
                    # edges.append((edge[0], edge[1], edge_exp_map[edge[2]['gt']]))
                    graph_viz.add_edge(edge[0], edge[1], color='black', weight=edge_weight_map[edge[2]['label']])
                
    else:
        for edge in nx_graph.edges.data():
            if ( edge[0], edge[1], EDGE_CLS[edge[2]['label']] ) not in edges :
                graph_viz.add_edge(edge[0], edge[1], color='blue', weight=edge_weight_map[edge[2]['label']])    
    
    # print("Edges: \n")
    # print(edges)
    
    # remove nodes without any color coding from the graph
    graph_viz.remove_nodes_from(list(nx.isolates(graph_viz)))
    
    # graph_viz.add_edges_from((u, v, {"type": label}) for u, v, label in edges)
    print(graph_viz.number_of_edges())
    print("\n\nNODES: \n")
    print(graph_viz.nodes.data())
    print("\n\nEDGES: \n")
    print(graph_viz.edges.data())
    
    return graph_viz


# format plot

def plot_graph(graph_viz, node_color_coding, title, graph_idx, pred_class, plot_save_path):

    plt.figure(1, figsize=(15, 10), dpi=60)

    # set baseline network plot options
    # base_options = dict(with_labels=False, edgecolors="black", node_size=300)

    # define layout of position
    # pos = nx.spring_layout(graph_viz, seed=7482934)
    pos = nx.spring_layout(graph_viz)
    

    # set node's color as we define previously  
    node_colors = [node_color_coding[str(n)]["color"] for n in graph_viz.nodes()]
    # node_colors = [d["color"] for _, d in graph_viz.nodes(data=True)]
    edge_colors = nx.get_edge_attributes(graph_viz, "color").values()
    edge_weights = nx.get_edge_attributes(graph_viz, "weight").values()
    print(node_colors)
    

    # edge_type_visual_weight_lookup = {"single": 2, "double": 4, "triple": 6, "single_exp": 8, "double_exp": 10, "triple_exp": 12}
    # edge_weights = [edge_type_visual_weight_lookup[d["type"]] for _, _, d in graph_viz.edges(data=True)]

    # edge_colors = [edge_color_coding[i] for i in range(graph_0.num_edges)]
    
    # add node's color to the plot


    # nx.draw_networkx(graph_viz, pos, node_color=node_colors, width=edge_weights, edge_color="black", **base_options)
    # nx.draw_shell(graph_viz, node_color=node_colors, width=edge_weights, edge_color="black", **base_options)
    nx.draw_networkx(graph_viz, pos, node_color = node_colors, width=list(edge_weights), edge_color=edge_colors, with_labels=True, font_weight='bold', node_size=300)
    # plt.show()
    # save_path = ""
    cat = 0
    if (pred_class == 0):
        save_path = plot_save_path + "Class_0/"
        cat = 0
    elif (pred_class == 1):
        # save_path = "./Exp_graph_plots/Class_1/"
        save_path = plot_save_path + "Class_1/"
        cat = 1
        
    
        
    plt.title(title + ': graph_idx:' + str(graph_idx) + ', class:' + str(cat) , fontsize=20)
    plt.savefig(save_path + 'graph_exp_' + str(graph_idx) + '_class_' + str(cat) + '.png')
    print("Saved graph plot for graph idx: ", graph_idx)
    plt.close()
    
def plot_cluster_representative(graph_viz, node_color_coding, title, cluster_no, pred_class, plot_save_path):
    
    plt.figure(1, figsize=(15, 10), dpi=60)
    pos = nx.spring_layout(graph_viz)

    # set node's color as we define previously  
    node_colors = [node_color_coding[str(n)]["color"] for n in graph_viz.nodes()]
    # print(node_colors)
    # node_elem_labels = [color_element_map[node_color_coding[str(n)]["color"]] for n in graph_viz.nodes()]
    node_elem_labels = {}
    for n in graph_viz.nodes():
        node_elem_labels[n] = COLOR_TO_NODE_MAP[node_color_coding[str(n)]["color"]]

    edge_colors = nx.get_edge_attributes(graph_viz, "color").values()
    edge_weights = nx.get_edge_attributes(graph_viz, "weight").values()
    
    # label nodes according to their color coding
    node_labels = nx.get_node_attributes(graph_viz, "label")
    
    nx.draw_networkx(graph_viz, pos, node_color = node_colors, width=list(edge_weights), edge_color=edge_colors, font_weight='bold', node_size=350, labels=node_elem_labels, with_labels=True)

    cat = 0
    if (pred_class == 0):
        cat = 0
    elif (pred_class == 1):
        cat = 1

    plt.title(title + ': Cluster_no:' + str(cluster_no) + ', class:' + str(cat) , fontsize=20)
    # plt.savefig(plot_save_path + 'factual_cluster_repr_' + str(cluster_no) + '_class_' + str(cat) + '.png')
    plt.savefig(plot_save_path + title + '_class_' + str(cat) + '.png')
    print("Saved graph plot for cluster_no: ", cluster_no)
    plt.close()