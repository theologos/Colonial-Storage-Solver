import pandas as pd
import numpy as num
import matplotlib.pyplot as plt
import gurobipy as gp
from   gurobipy import GRB
import time
import math
import csv
import os
import warnings
import re
import collections
import dash
import json
import sys
import io
from datetime import datetime, timedelta
from dash import Dash, dcc, html, Input, Output, State, ALL
import dash_bootstrap_components as dbc
from mip import Model
from pulp import LpProblem

pd.options.mode.chained_assignment = None

from class_OptimizationModel import OptimizationModel
from class_data import DataCycleLoader
from class_data import DataOptimizations

data_cycle_loader = DataCycleLoader();
data_optimizations = DataOptimizations(data_cycle_loader);


# Assuming the rest of your script is the same...

if __name__ == "__main__":
    if len(sys.argv) == 3:
        # Capture inputs from command line arguments
        arg1 = sys.argv[1]
        arg2 = sys.argv[2]

        # Example usage of the arguments
        inputs = data_optimizations.getOptimization(arg1, arg2).inputs # For example "ATJ" "031"
        
        ID = 1                
        # Read the Tanks_to_use
        df_read = pd.read_csv('input_optimization/tanks.csv')
        inputs['Tanks_to_use'] = df_read['Tanks'].tolist()
        
        # Read the flow_constraints. Example: [{'Tank': 310, 'Inbound': 2, 'Outbound': 2}, {'Tank': 311, 'Inbound': 2, 'Outbound': 2}]
        df_read = pd.read_csv('input_optimization/flowTanks.csv')
        inputs['flow_constraints'] = [{'Tank': row['Tank'], 'Inbound': row['Inbound'], 'Outbound': row['Outbound']} for index, row in df_read.iterrows()]

        # Read additional variables. Example: input["flow_constraints_univ_1"] = 3, input["flow_constraints_univ_2"] = 3
        df_read = pd.read_csv('input_optimization/misc.csv')
        inputs.update({row['Variable']: int(row['Value']) for index, row in df_read.iterrows()})

        # Set objective
        inputs['objective'] = "selection1"
        
        # Call the optimization model
        ret = OptimizationModel.model_stage3(ID, inputs)

