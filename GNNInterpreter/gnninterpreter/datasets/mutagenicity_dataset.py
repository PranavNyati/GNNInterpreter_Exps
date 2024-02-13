import networkx as nx
import pandas as pandas
import torch_geometric as pyg
import matplotlib.pyplot as plt
import pandas as pd

from .base_graph_dataset import BaseGraphDataset
from .utils import default_ax, unpack_G

class MutagenicityDataset(BaseGraphDataset):
    
    NODE_CLS = {

        0: 'C',
        1: 'O',
        2: 'Cl',
        3: 'H',
        4: 'N',
        5: 'F',
        6: 'Br',
        7: 'S',
        8: 'P',
        9: 'I',
        10: 'Na',
        11: 'K',
        12: 'Li',
        13: 'Ca',
    }
    
    # REV_NODE_MAP = {v: k for k, v in NODE_CLS.items()}
    REV_NODE_MAP = {
        'C': 0,
        'O': 1,
        'Cl': 2,
        'H': 3,
        'N': 4,
        'F': 5,
        'Br': 6,
        'S': 7,
        'P': 8,
        'I': 9,
        'Na': 10,
        'K': 11,
        'Li': 12,
        'Ca': 13
    }
    
    NODE_COLOR = {
        0: 'red',
        1: 'blue',
        2: 'green',
        3: 'yellow',
        4: 'orange',
        5: 'purple',
        6: 'pink',
        7: 'brown',
        8: 'black',
        9: 'gray',
        10: 'cyan',
        11: 'magenta',
        12: 'olive',
        13: 'lime'
    }

    color_element_map = {
        'red': 'C',
        'blue': 'O',
        'green': 'Cl',
        'yellow': 'H',
        'orange': 'N',
        'purple': 'F',
        'pink': 'Br',
        'brown': 'S',
        'black': 'P',
        'gray': 'I',
        'cyan': 'Na',
        'magenta': 'K',
        'olive': 'Li',
        'lime': 'Ca'
    }

    EDGE_CLS = {
        0: 'single',
        1: 'double',
        2: 'triple'
    }

    EDGE_WIDTH = {
        0: 3,
        1: 6,
        2: 9
    }

    # edge_exp_map = {
    #     0.0: 'single_exp',
    #     1.0: 'double_exp',
    #     2.0: 'triple_exp'
    # }

    # creating a color map for the edges
    EDGE_COLOR = {
        0: 'black',
        1: 'red',
        2: 'blue',
    }

    GRAPH_CLS = {
        0: 'mutagen',
        1: 'nonmutagen',
    }
    
    
    def __init__(self, *,
                 name='Mutagenicity',
                 url='https://www.chrsmrrs.com/graphkerneldatasets/Mutagenicity.zip',
                 **kwargs):
        self.url = url
        super().__init__(name=name, **kwargs)
        
    @property
    def raw_file_names(self):
        return ['Mutagenicity/Mutagenicity_A.txt',
                'Mutagenicity/Mutagenicity_graph_indicator.txt',
                'Mutagenicity/Mutagenicity_graph_labels.txt',
                'Mutagenicity/Mutagenicity_edge_labels.txt',
                'Mutagenicity/Mutagenicity_node_labels.txt']

    def download(self):
        pyg.data.download_url(self.url, self.raw_dir)
        pyg.data.extract_zip(f'{self.raw_dir}/Mutagenicity.zip', self.raw_dir)
        
    def generate(self):
        edges = pd.read_csv(self.raw_paths[0], header=None).to_numpy(dtype=int) - 1
        graph_idx = pd.read_csv(self.raw_paths[1], header=None)[0].to_numpy(dtype=int) - 1
        graph_labels = pd.read_csv(self.raw_paths[2], header=None)[0].to_numpy(dtype=int) 
        edge_labels = pd.read_csv(self.raw_paths[3], header=None)[0].to_numpy(dtype=int)
        node_labels = pd.read_csv(self.raw_paths[4], header=None)[0].to_numpy(dtype=int)                                
        super_G = nx.Graph(edges.tolist(), label=graph_labels)
        nx.set_node_attributes(super_G, dict(enumerate(node_labels)), name='label')
        nx.set_node_attributes(super_G, dict(enumerate(graph_idx)), name='graph')
        nx.set_edge_attributes(super_G, dict(zip(zip(*edges.T), edge_labels)), name='label')
        return unpack_G(super_G)


    # TODO: use EDGE_WIDTH
    @default_ax
    def draw(self, G, pos=None, label=False, ax=None, path=None):
        plt.figure(1, figsize=(15, 10), dpi=60)
        # pos = pos or nx.kamada_kawai_layout(G)
        pos = pos or nx.spring_layout(G)
        nx.draw_networkx_nodes(G, pos,
                               ax=ax,
                               nodelist=G.nodes,
                               node_size=500,
                               node_color=[
                                   self.NODE_COLOR[G.nodes[v]['label']]
                                   for v in G.nodes
                               ])
        if label:
            nx.draw_networkx_labels(G, pos,
                                    ax=ax,
                                    labels={
                                        v: self.NODE_CLS[G.nodes[v]['label']]
                                        for v in G.nodes
                                    })
        nx.draw_networkx_edges(G.subgraph(G.nodes), pos, ax=ax, width=6)
        
        # plt.savefig(path)
    
    def draw_graph(self, G):
        
        graph_viz = nx.Graph()
        
        for v in G.nodes():
            # graph_viz.add_node( (v, {'label': self.NODE_CLS[G.nodes[v]['label']], 'color': self.NODE_COLOR[G.nodes[v]['label']]} ) )4
            graph_viz.add_node( v, label= self.NODE_CLS[G.nodes[v]['label']], color= self.NODE_COLOR[G.nodes[v]['label']] )
        print("Graph Viz info: ")
        print("No of nodes: ", graph_viz.number_of_nodes())
        print("Nodes: ", graph_viz.nodes(data=True))
                    
        for u, v in G.edges():
            graph_viz.add_edge( u, v, label= self.EDGE_CLS[G[u][v]['label']], color= self.EDGE_COLOR[G[u][v]['label']], width= self.EDGE_WIDTH[G[u][v]['label']])
        
        graph_viz.remove_nodes_from(list(nx.isolates(graph_viz)))
        
        print("No of edges: ", graph_viz.number_of_edges())
        print("Edges: ", graph_viz.edges(data=True))
        
        
        plt.figure(1, figsize=(15, 10), dpi=60)

        # set baseline network plot options
        # base_options = dict(with_labels=False, edgecolors="black", node_size=300)

        # define layout of position
        # pos = nx.spring_layout(graph_viz, seed=7482934)
        pos = nx.spring_layout(graph_viz)
        # nx.draw_networkx(graph_viz, pos, node_color = [node[1]['color'] for node in graph_viz.nodes(data=True)], node_size=500, edge_color = [edge[2]['color'] for edge in graph_viz.edges(data=True)], width = [edge[2]['width'] for edge in graph_viz.edges(data=True)], with_labels=True)
        nx.draw_networkx(graph_viz, pos, node_color = [node[1]['color'] for node in graph_viz.nodes(data=True)],  edge_color = [edge[2]['color'] for edge in graph_viz.edges(data=True)], width = [edge[2]['width'] for edge in graph_viz.edges(data=True)], with_labels=True, font_weight='bold', node_size=300)
        
    def process(self):
        super().process()
        