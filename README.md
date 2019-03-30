community_detection

implement a algorithm which represent each node in the network as a low-dimension vector and then clustering on the learned nodes representations to detect the communities. 

## graph data format

Every network dataset need include three files:

* .edgelist file:  each line represent one edge in the network
* .feature file: each line represent one node's attribute vector
* .label file: each line show the community node belong 

## model parameters configuration

Tuning the model parameters in config.py.

## output

The output node embeddings are saved in the .embedding file.

## visualization

The learned node embeddings can be used for network visualization. Nodes in the network can be shown as matplotlib scatter after reduce the dimension of  embeddings to 2d by using t-sne package. Nodes similar with each other will also close in embedding space. There is an example as follow:

![](https://github.com/xiexiaomiao/comunity_detection/blob/master/img/come.jpeg)