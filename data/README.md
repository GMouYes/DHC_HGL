# We include an example of data from Extrasensory. It sampled 10% of instances from 3 users. Please unzip the train/test/valid folder before the test run.
## nodeInit_ruled.npy and edgeInit_ruled.npy contains initial node and edge features.
## The adj_ruled.npy indicates the adjacency matrix of the graph. Specifically, adj_ruled_u_pp.npy, adj_ruled_u_a.npy and adj_ruled_u_pp_a.npy contains the adjacency matrix for three sub-hypergraph, namely G<sub>{u,a}</sub>, G<sub>{u,pp}</sub>, and G<sub>{u,pp,a}</sub>
## count_ruled.npy indicates edge weight. Similar to the adjacency matrix, three files also indicate hyperedge weights for the three sub-hypergraphs.
## freature_ruled.npy and y_ruled.npy are the instances (X) and labels (Y), where X is the handcrafted features we extracted from raw sensor signals and Y contains user, phone placement, and activity labels. 
## And mask_expanded_ruled.npy is the instance-pair weight of the label. The weight of the missing label is set to 0 so that it won't contribute to the loss.
