# Concise graph feature analysis with heatmaps

A heatmap shows how a graph feature changes given different time windows.
The heatmap offers a more rigorous analysis of certain graph features
to understand whether it is reliable to signal important market events.


## Baseline version


Launch ``heatmap.py`` to compute MSTs features

examples:
- $: python3 heatmap.py --csv ../data/Ita-2019-2020-July_classified_small.csv  --out out/rlt.json
- $: python3 heatmap.py --csv ../data/Ita-2019-2020-July_classified_small.csv  --presenceTreshold 0.8 --overlapThreshold 0.8  --ifIncludeNegativeEdges false  --winSizeStep 3 --edgeThreshold 0.1 --out out/tmp.json

## Visualization

$: python3 showHeatmap.py

Update variable ``inputFile`` for file location and variable ``topic`` for the type of graph feature.


## Faster heatmap computation

Launch ``fastHeatmap.py``. Currently it only supports first degree score (degree distribution).

example 
$: python3 fastHeatmap.py --csv ../data/Ita-2019-2020-July_classified_small.csv  --out out/rlt.json


### Note, compile the binary version of heatmap computation if the binary package is broken
python3 setup.py build_ext --inplace