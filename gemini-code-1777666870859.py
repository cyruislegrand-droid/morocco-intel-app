import networkx as nx

def build_actor_network(df):
    G = nx.Graph()
    for _, row in df.iterrows():
        actors = row['actors']
        for i in range(len(actors)):
            for j in range(i + 1, len(actors)):
                G.add_edge(actors[i], actors[j], weight=1)
    return G