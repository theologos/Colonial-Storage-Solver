import pandas as pd
import numpy as num
from class_data import DataLocation;
from class_data import DataCycleLoader;
from class_data import DataOptimization;
from class_data import DataOptimizations;
#------------------------------------------------------------------------------------------
#
#
#data_location = DataLocation("ATJ")
data_cycle_loader = DataCycleLoader();
data_optimizations = DataOptimizations(data_cycle_loader);
#inputs = data_optimizations.getOptimization('ATJ', '031').inputs # For example "ATJ" "031"

object = data_optimizations.getOptimization('ATJ', '041');
#print(object.index['tank'][0:10])
#print(object.topo_x)




combined_dict = {**object.topo_i, **object.topo_o}
topo_x = {}
for outer_key, middle_dict in combined_dict.items():
    for middle_key, inner_dict in middle_dict.items():
        for inner_key in inner_dict.keys():
            topo_x[outer_key] = inner_key
            break  # Since we only need the first key, we can break after finding it

#print(topo_x)

a = [(tank, topo_x[tank], time) for tank in topo_x for time in [1,2,3]]
print(a)