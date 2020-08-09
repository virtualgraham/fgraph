CALL gds.graph.create(
    'graph',
    'Observation',
    'ADJACENT'
)
YIELD graphName, nodeCount, relationshipCount, createMillis;


CALL gds.louvain.write('graph', { writeProperty: 'group2', maxLevels: 1700, maxIterations: 1700 }) YIELD communityCount, modularity, modularities