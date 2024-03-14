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
object.print_details()
