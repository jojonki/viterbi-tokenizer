# Use Graham's example.
# http://www.phontron.com/slides/nlp-programming-ja-03-ws.pdf

INF = 1e6
edge_list = [
    None, # e0
    { # e1
        'id': 1,
        'score': 2.5,
        'begin_node_id': 0,
        'end_node_id': 1
    },
    { # e2
        'id': 2,
        'score': 1.4,
        'begin_node_id': 0,
        'end_node_id': 2
    },
    { # e3
        'id': 3,
        'score': 4.0,
        'begin_node_id': 1,
        'end_node_id': 2
    },
    { # e4
        'id': 4,
        'score': 2.1,
        'begin_node_id': 1,
        'end_node_id': 3
    },
    { # e5
        'id': 5,
        'score': 2.3,
        'begin_node_id': 2,
        'end_node_id': 3
    }
]

best_edges = []

node_list = [
    { # node 0
        'id': 0,
        'best_score': None,
        'best_edge': None,
        'incoming_edges': [],
    },
    { # node 1
        'id': 1,
        'best_score': None,
        'best_edge': None,
        'incoming_edges': [1],
    },
    { # node 2
        'id': 2,
        'best_score': None,
        'best_edge': None,
        'incoming_edges': [2, 3],
    },
    { # node 3
        'id': 3,
        'best_score': None,
        'best_edge': None,
        'incoming_edges': [4, 5],
    }
]


def forward():
    node_list[0]['best_score'] = 0
    for nid in range(1, len(node_list)):
        node = node_list[nid]
        node['best_score'] = INF
        for edge_id in node['incoming_edges']:
            edge = edge_list[edge_id]
            edge_begin_node = node_list[edge['begin_node_id']]
            score = edge_begin_node['best_score'] + edge['score']
            if score < node['best_score']:
                node['best_score'] = score
                node['best_edge'] = edge_list[edge_id]
        best_edges.append(node['best_edge'])

    print('Node cost results:')
    for n in node_list:
        print(n)
    print('best_edges:', best_edges)


def backward():
    next_edge = best_edges[-1]
    best_path = []
    while next_edge is not None:
        begin_node = node_list[next_edge['begin_node_id']]
        best_path.append(next_edge['id'])
        next_edge = begin_node['best_edge']
    best_path = best_path[::-1]
    print('best_path:', best_path)


if __name__ == '__main__':
    forward()
    backward()
