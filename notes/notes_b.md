# TODO
    [ ] Use multiple window sizes while allowing adjacencies between different window sizes
    [ ] Try Hog descriptors
    [x] Try cranking up identical distance to be neighbor distance
        [ ] keep running mean or max for integrated feats
        [ ] What happens when more than one nn is within identical range? Use multiple nodes as current node, or continue just using the nearest neighbor?
    [ ] Try removing all nodes with 2 or less edges
    [x] When expanding walk size to discover community, the first step 1, should just be the node without a walk. Then a 2 step walk, 4 step walk....
