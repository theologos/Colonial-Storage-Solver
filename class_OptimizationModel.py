import pandas as pd
import math
import csv
import numpy as num
import matplotlib.pyplot as plt
import gurobipy as gp
from   gurobipy import GRB
import time

from class_analysis import DataAnalysis

class OptimizationModel:

    # @staticmethod
    # def function_data(ID, inputs):
        
    #     i_index, o_index, x, i, o, p, q, T = (
    #         inputs['index']['i'],
    #         inputs['index']['o'],
    #         inputs['x'],
    #         inputs['i'],
    #         inputs['o'],
    #         inputs['p'],
    #         inputs['q'],
    #         inputs['T']
    #     )

    #     data = []
    #     tups = [tup for tup in i_index]
    #     for tup in tups:
    #         tank = tup[0]; line = tup[1]; prod = tup[2]; j = tup[3];
    #         if p[tank, line, prod, j].X > 0 and i[tank, line, prod, j].X > 0:
    #             row = [j, tank, line, prod, x[tank, prod, j].X, i[tank, line, prod, j].X]
    #             data.append(row)

    #     tups = [tup for tup in o_index]
    #     for tup in tups:
    #         tank = tup[0]; line = tup[1]; prod = tup[2]; j = tup[3];
    #         if q[tank, line, prod, j].X > 0 and o[tank, line, prod, j].X > 0:
    #             row = [j, tank, line, prod, x[tank, prod, j].X, o[tank, line, prod, j].X]
    #             data.append(row)

    #     data = pd.DataFrame (data, columns = ['Time', 'Tank', 'Line', 'Product', 'x', 'io'])    
        
    #     data1 = data.groupby(['Line', 'Product']).agg({'Time':'min'}).reset_index().rename(columns = {"Time":"Time_min"})
    #     data2 = data.groupby(['Line', 'Product']).agg({'Time':'max'}).reset_index().rename(columns = {"Time":"Time_max"})
    #     data = data1.merge(data2, on=["Line", "Product"])

    #     def order(df):
    #         if (df['Product'] == 'A'):
    #             return 1
    #         elif (df['Product'] == 'D'):
    #             return 2
    #         elif (df['Product'] == '54'):
    #             return 3
    #         elif (df['Product'] == '62'):
    #             return 4
    #     data['Order'] = data.apply(order, axis = 1)

    #     data = data.sort_values(['Line', 'Order'], ascending = [True, True])
    #     #data
    #     data['Time_min_1'] = data.groupby(['Line'])['Time_max'].shift(1)
    #     data['Time_max_1'] = data.groupby(['Line'])['Time_min'].shift(-1)
    #     data['Time_min_1'] = data['Time_min_1'].fillna(data['Time_min'])
    #     data['Time_max_1'] = data['Time_max_1'].fillna(data['Time_max'])

    #     data['Time_min_f'] = (data['Time_min'] + data['Time_min_1'])/2
    #     data['Time_max_f'] = (data['Time_max'] + data['Time_max_1'])/2

    #     def fun(df):
    #         return math.ceil(df['Time_min_f'])
    #     data['Time_min_f'] = data.apply(fun, axis = 1)

    #     def fun(df):
    #         return math.ceil(df['Time_max_f'])
    #     data['Time_max_f'] = data.apply(fun, axis = 1)

    #     data['Order'] = data[['Line', 'Order']].groupby(['Line']).rank(ascending=False)
    #     def fun(df):
    #         if df['Order'] != 1:
    #             return df['Time_max_f'] - 1
    #         else:
    #             return df['Time_max_f']
    #     data['Time_max_f'] = data.apply(fun, axis = 1)

    #     def fun(df):
    #         if df['Order'] == 1:
    #             return T-1
    #         else:
    #             return df['Time_max_f']
    #     data['Time_max_f'] = data.apply(fun, axis = 1)

    #     def line(df):
    #         if (df['Line'] in [1]):
    #             return '01'
    #         elif (df['Line'] in [2]):
    #             return '02'
    #         else:
    #             return str(df['Line'])

    #     data['Line'] = data.apply(line, axis = 1)
    #     data.to_csv("results/opt_timeline_" + str(ID) + ".csv")

    #     return (data)

    @staticmethod
    def model_baseline_const1(model, inputs):
        
        """
        Adds constraints to the given mathematical optimization `model` based on the provided `inputs`.

        This function adds flow balance constraints.

        Parameters:
            model (OptimizationModel): The mathematical optimization model to which constraints are added.
            inputs (dict): A dictionary containing input data required to create constraints.
                The dictionary should have the following keys:
                    - 'index': A dictionary containing information about 'x', 'i', and 'o' indexes.
                        It should have the following keys:
                            - 'x': A list of tuples representing the 'x' index with format (tank, prod, time, ...)
                            - 'i': A list of tuples representing the 'i' index with format (tank, line, prod, time, ...)
                            - 'o': A list of tuples representing the 'o' index with format (tank, line, prod, time, ...)
                    - 'x': The optimization variable 'x' representing tank volume in the model.
                    - 'i': The optimization variable 'i' representing inflow in the model.
                    - 'o': The optimization variable 'o' representing outflow in the model.
                    - 'p': The optimization variable 'p' representing production in the model.
                    - 'q': The optimization variable 'q' representing consumption in the model.
                    - 'CycleVolIn2': A dictionary containing cycle volume constraints for inflow operations.
                    - 'CycleVolOut2': A dictionary containing cycle volume constraints for outflow operations.
                    - 'CycleVolExist': A dictionary containing cycle volume constraints for existing tanks.
                    - 'Tanks': A list of tanks to consider in the constraints.
                    - 'T': The total number of time steps in the model.
                    - 'Capacity': A dictionary containing the capacity constraints for each tank.

        Returns:
            OptimizationModel: The `model` with added constraints.

        Example:
            # Assume 'model' and 'inputs' are defined.
            model = function_model_const1(model, inputs)

        Note:
            - The 'x', 'i', 'o', 'p', and 'q' optimization variables must be previously defined in the model before calling this function.
        """    
            
            
        x_index, i_index, o_index, x, i, o, p, q, CycleVolIn2, CycleVolOut2, CycleVolExist, Tanks, T, Capacity = (
            inputs['index']['x'],
            inputs['index']['i'],
            inputs['index']['o'],
            inputs['x'],
            inputs['i'],
            inputs['o'],
            inputs['p'],
            inputs['q'],
            inputs['CycleVolIn2'],
            inputs['CycleVolOut2'],
            inputs['CycleVolExist'],
            inputs['Tanks'],
            inputs['T'],
            inputs['Capacity']
        )

        #-------------------------------------------------------------------------------------------------------------
        # Initial Volume condition
        #
        for tank in CycleVolExist:
            for prod in CycleVolExist[tank]:
                model.addConstr(x[tank, prod, 0] == CycleVolExist[tank][prod])     
            
        #-------------------------------------------------------------------------------------------------------------    
        # Pruduct does not exceed volume at any given time
        #
        lst = list(set([(t[0],t[2]) for t in x_index])) 
        for tup in lst:
            tank = tup[0]; time = tup[1];
            model.addConstr(x.sum(tank, '*', time) <= Capacity[tank] + 0)         

        #-------------------------------------------------------------------------------------------------------------    
        # Flow Balance constraints
        #    
        for tank in Tanks:
            prod    = [tup[1] for tup in x_index if tup[0] == tank and tup[2] == 0]
            lineIn  = [tup[1] for tup in i_index if tup[0] == tank and tup[2] == prod[0] and tup[3] == 0]
            lineOut = [tup[1] for tup in o_index if tup[0] == tank and tup[2] == prod[0] and tup[3] == 0]
            prod    = prod[0]; 

            #print(tank, line, prod, lineIn, lineOut)   
            if (len(lineIn) == 0):
                if (len(lineOut) == 1): 
                    line1 = lineOut[0]
                    for j in list(range(T-1)):
                        model.addConstr(x[tank, prod, j + 1] == x[tank, prod, j] - o[tank, line1, prod, j])

                if (len(lineOut) == 2):
                    line1 = lineOut[0]; line2 = lineOut[1]
                    for j in list(range(T-1)):
                        model.addConstr(x[tank, prod, j + 1] == x[tank, prod, j] - o[tank, line1, prod, j] - o[tank, line2, prod, j])

                if (len(lineOut) == 3):
                    line1 = lineOut[0]; line2 = lineOut[1]; line3 = lineOut[2]
                    for j in list(range(T-1)):
                        model.addConstr(x[tank, prod, j + 1] == x[tank, prod, j] - o[tank, line1, prod, j] - o[tank, line2, prod, j] - o[tank, line3, prod, j])

                if (len(lineOut) == 4):
                    line1 = lineOut[0]; line2 = lineOut[1]; line3 = lineOut[2]; line4 = lineOut[3]
                    for j in list(range(T-1)):
                        model.addConstr(x[tank, prod, j + 1] == x[tank, prod, j] - o[tank, line1, prod, j] - o[tank, line2, prod, j] - o[tank, line3, prod, j] - o[tank, line4, prod, j])
                        
                if (len(lineOut) == 5):
                    line1 = lineOut[0]; line2 = lineOut[1]; line3 = lineOut[2]; line4 = lineOut[3]; line5 = lineOut[4]
                    for j in list(range(T-1)):
                        model.addConstr(x[tank, prod, j + 1] == x[tank, prod, j] - o[tank, line1, prod, j] - o[tank, line2, prod, j] - o[tank, line3, prod, j] - o[tank, line4, prod, j] - o[tank, line5, prod, j])
                        
                if (len(lineOut) == 6):
                    line1 = lineOut[0]; line2 = lineOut[1]; line3 = lineOut[2]; line4 = lineOut[3]; line5 = lineOut[4]; line6 = lineOut[5]
                    for j in list(range(T-1)):
                        model.addConstr(x[tank, prod, j + 1] == x[tank, prod, j] - o[tank, line1, prod, j] - o[tank, line2, prod, j] - o[tank, line3, prod, j] - o[tank, line4, prod, j] - o[tank, line5, prod, j] - o[tank, line6, prod, j])
                    
                if (len(lineOut) == 7):
                    line1 = lineOut[0]; line2 = lineOut[1]; line3 = lineOut[2]; line4 = lineOut[3]; line5 = lineOut[4]; line6 = lineOut[5]; line7 = lineOut[6]
                    for j in list(range(T-1)):
                        model.addConstr(x[tank, prod, j + 1] == x[tank, prod, j] - o[tank, line1, prod, j] - o[tank, line2, prod, j] - o[tank, line3, prod, j] - o[tank, line4, prod, j] - o[tank, line5, prod, j] - o[tank, line6, prod, j] - o[tank, line7, prod, j]) 
                        
                if (len(lineOut) == 8):
                    line1 = lineOut[0]; line2 = lineOut[1]; line3 = lineOut[2]; line4 = lineOut[3]; line5 = lineOut[4]; line6 = lineOut[5]; line7 = lineOut[6]; line8 = lineOut[7]
                    for j in list(range(T-1)):
                        model.addConstr(x[tank, prod, j + 1] == x[tank, prod, j] - o[tank, line1, prod, j] - o[tank, line2, prod, j] - o[tank, line3, prod, j] - o[tank, line4, prod, j] - o[tank, line5, prod, j] - o[tank, line6, prod, j] - o[tank, line7, prod, j] - o[tank, line8, prod, j])    

            if (len(lineIn) == 1):
                line = lineIn[0]
                if (len(lineOut) == 0): 
                    for j in list(range(T-1)):
                        model.addConstr(x[tank, prod, j + 1] == x[tank, prod, j] + i[tank, line, prod, j])

                if (len(lineOut) == 1): 
                    line1 = lineOut[0]
                    for j in list(range(T-1)):
                        model.addConstr(x[tank, prod, j + 1] == x[tank, prod, j] + i[tank, line, prod, j] - o[tank, line1, prod, j])

                if (len(lineOut) == 2):
                    line1 = lineOut[0]; line2 = lineOut[1]
                    for j in list(range(T-1)):
                        model.addConstr(x[tank, prod, j + 1] == x[tank, prod, j] + i[tank, line, prod, j] - o[tank, line1, prod, j] - o[tank, line2, prod, j])

                if (len(lineOut) == 3):
                    line1 = lineOut[0]; line2 = lineOut[1]; line3 = lineOut[2]
                    for j in list(range(T-1)):
                        model.addConstr(x[tank, prod, j + 1] == x[tank, prod, j] + i[tank, line, prod, j] - o[tank, line1, prod, j] - o[tank, line2, prod, j] - o[tank, line3, prod, j])

                if (len(lineOut) == 4):
                    line1 = lineOut[0]; line2 = lineOut[1]; line3 = lineOut[2]; line4 = lineOut[3]
                    for j in list(range(T-1)):
                        model.addConstr(x[tank, prod, j + 1] == x[tank, prod, j] + i[tank, line, prod, j] - o[tank, line1, prod, j] - o[tank, line2, prod, j] - o[tank, line3, prod, j] - o[tank, line4, prod, j])
                    
                if (len(lineOut) == 5):
                    line1 = lineOut[0]; line2 = lineOut[1]; line3 = lineOut[2]; line4 = lineOut[3]; line5 = lineOut[4]
                    for j in list(range(T-1)):
                        model.addConstr(x[tank, prod, j + 1] == x[tank, prod, j] + i[tank, line, prod, j] - o[tank, line1, prod, j] - o[tank, line2, prod, j] - o[tank, line3, prod, j] - o[tank, line4, prod, j] - o[tank, line5, prod, j])
                        
                if (len(lineOut) == 6):
                    line1 = lineOut[0]; line2 = lineOut[1]; line3 = lineOut[2]; line4 = lineOut[3]; line5 = lineOut[4]; line6 = lineOut[5]
                    for j in list(range(T-1)):
                        model.addConstr(x[tank, prod, j + 1] == x[tank, prod, j] + i[tank, line, prod, j] - o[tank, line1, prod, j] - o[tank, line2, prod, j] - o[tank, line3, prod, j] - o[tank, line4, prod, j] - o[tank, line5, prod, j] - o[tank, line6, prod, j]) 
                        
                if (len(lineOut) == 7):
                    line1 = lineOut[0]; line2 = lineOut[1]; line3 = lineOut[2]; line4 = lineOut[3]; line5 = lineOut[4]; line6 = lineOut[5]; line7 = lineOut[6]
                    for j in list(range(T-1)):
                        model.addConstr(x[tank, prod, j + 1] == x[tank, prod, j] + i[tank, line, prod, j] - o[tank, line1, prod, j] - o[tank, line2, prod, j] - o[tank, line3, prod, j] - o[tank, line4, prod, j] - o[tank, line5, prod, j] - o[tank, line6, prod, j] - o[tank, line7, prod, j])             
                        
                if (len(lineOut) == 8):
                    line1 = lineOut[0]; line2 = lineOut[1]; line3 = lineOut[2]; line4 = lineOut[3]; line5 = lineOut[4]; line6 = lineOut[5]; line7 = lineOut[6]; line8 = lineOut[7]
                    for j in list(range(T-1)):
                        model.addConstr(x[tank, prod, j + 1] == x[tank, prod, j] + i[tank, line, prod, j] - o[tank, line1, prod, j] - o[tank, line2, prod, j] - o[tank, line3, prod, j] - o[tank, line4, prod, j] - o[tank, line5, prod, j] - o[tank, line6, prod, j] - o[tank, line7, prod, j] - o[tank, line8, prod, j])
        
        return (model)

    @staticmethod
    def model_baseline_const2(model, inputs):
        """
        Adds constraints to the given mathematical optimization `model` based on the provided `inputs`.

        This function adds constraints to the optimization model to ensure that each Line can only accept Product from one Tank at any given time.

        Parameters:
            model (OptimizationModel): The mathematical optimization model to which constraints are added.
            inputs (dict): A dictionary containing input data required to create constraints.
                The dictionary should have the following keys:
                    - 'index': A dictionary containing information about 'i' and 'o' indexes.
                        It should have the following keys:
                            - 'i': A list of tuples representing the 'i' index with format (line, prod, time, ...)
                            - 'o': A list of tuples representing the 'o' index with format (line, prod, time, ...)
                    - 'p': The optimization variable 'p' in the model.
                    - 'q': The optimization variable 'q' in the model.

        Returns:
            OptimizationModel: The `model` with added constraints.

        Example:
            # Assume 'model' and 'inputs' are defined.
            model = function_model_const1(model, inputs)

        Note:
            - The 'p' and 'q' optimization variables must be previously defined in the model before calling this function.
        """
        i_index = inputs['index']['i'] 
        o_index = inputs['index']['o']
        p = inputs['p'] 
        q = inputs['q']
        
        lst = list(set([t[1:4] for t in i_index]))            
        for tup in lst:
            line = tup[0]
            prod = tup[1]
            time = tup[2]
            model.addConstr(p.sum('*', line, prod, time) <= 1)   

        lst = list(set([t[1:4] for t in o_index]))            
        for tup in lst:
            line = tup[0]
            prod = tup[1]
            time = tup[2]
            model.addConstr(q.sum('*', line, prod, time) <= 1)

        return model
    
    @staticmethod
    def model_baseline_const3(model, inputs):
        
        """
        Adds constraints to the given mathematical optimization `model` based on the provided `inputs`.

        This function adds constraints to the optimization model to ensure
            - Inbound volume:
                Is about the volume indicated by "CycleVolIn2"
                The flow rate satisfies the bounds indicated by "Bounds"
            - Outbound volume
                Is about the volume indicated by "CycleVolOut2"
                The flow rate satisfies the bounds indicated by "Bounds"
            - Zero flows:
                Zero outflows for the state space that does not belong to CycleVolOut2

        Parameters:
            model (OptimizationModel): The mathematical optimization model to which constraints are added.
            inputs (dict): A dictionary containing input data required to create constraints.
                The dictionary should have the following keys:
                    - 'index': A dictionary containing information about 'i' and 'o' indexes.
                        It should have the following keys:
                            - 'i': A list of tuples representing the 'i' index with format (tank, line, prod, j, ...)
                            - 'o': A list of tuples representing the 'o' index with format (tank, line, prod, j, ...)
                    - 'i', 'p': The optimization variables representing inflow in the model.
                    - 'o', 'q': The optimization variables representing outflow in the model.
                    - 'CycleVolIn2': A dictionary containing cycle volume for inflow operations.
                    - 'CycleVolOut2': A dictionary containing cycle volume for outflow operations.
                    - 'Bounds': A dictionary containing line flow rate bounds.

        Returns:
            OptimizationModel: The `model` with added constraints.

        Example:
            # Assume 'model' and 'inputs' are defined.
            model = function_model_const3(model, inputs)

        Note:
            - The 'i', 'o', 'p', and 'q' optimization variables must be previously defined in the model before calling this function.
            - The function will create constraints for cycle volume constraints of inflow and outflow operations specified in 'CycleVolIn2' and 'CycleVolOut2'.
            - The function will create constraints based on bounds specified in 'Bounds' for both inflow and outflow operations.
            - If a line and product combination specified in 'CycleVolOut2' has no value (None), the corresponding outflow and consumption will be set to 0.
        """

        i_index, o_index, i, o, p, q, CycleVolIn2, CycleVolOut2, Bounds = (
            inputs['index']['i'],
            inputs['index']['o'],
            inputs['i'],
            inputs['o'],
            inputs['p'],
            inputs['q'],
            inputs['CycleVolIn2'],
            inputs['CycleVolOut2'],
            inputs['Bounds']
        )

        #-------------------------------------------------------------------------------------------------------------              
        # Inflow Constraints
        #
        for line in CycleVolIn2:
            for prod in CycleVolIn2[line]:
                model.addConstr(i.sum("*", line, prod, "*") >= CycleVolIn2[line][prod])
                model.addConstr(i.sum("*", line, prod, "*") <= CycleVolIn2[line][prod] + 30)
        
        lst = list(set([t[0:4] for t in i_index]))            
        for tup in lst:
            tank = tup[0]; line = tup[1]; prod = tup[2]; j = tup[3]
            model.addConstr(i[tank, line, prod, j] - p[tank, line, prod, j] * Bounds[line]["u"]  <= 0)
            model.addConstr(i[tank, line, prod, j] - p[tank, line, prod, j] * Bounds[line]["l"]  >= 0) 
            
        #-------------------------------------------------------------------------------------------------------------                        
        # Outflow constraints
        #
        for line in CycleVolOut2:
            for prod in CycleVolOut2[line]:
                model.addConstr(o.sum("*", line, prod, "*") >= CycleVolOut2[line][prod] )            
                model.addConstr(o.sum("*", line, prod, "*") <= CycleVolOut2[line][prod] + 10)            
        
        lst = list(set([t[0:4] for t in o_index]))            
        for tup in lst:
            tank = tup[0]; line = tup[1]; prod = tup[2]; j = tup[3]
            model.addConstr(o[tank, line, prod, j] - q[tank, line, prod, j] * Bounds[line]["u"] <= 0)
            model.addConstr(o[tank, line, prod, j] - q[tank, line, prod, j] * Bounds[line]["l"] >= 0) 

        #-------------------------------------------------------------------------------------------------------------                        
        # Zero outflows 
        #    
        lst = list(set([t[0:4] for t in o_index]))            
        for tup in lst:
            tank = tup[0]; line = tup[1]; prod = tup[2]; j = tup[3]
            value = CycleVolOut2.get(line, {}).get(prod)
            if value is None:
                model.addConstr(q[tank, line, prod, j] == 0)
                model.addConstr(o[tank, line, prod, j] == 0)
        
        return (model)

    @staticmethod
    def model_baseline_const3_v2(model, inputs):
    
        i_index, o_index, i, o, p, q, CycleVolIn2, CycleVolOut2, Bounds, CycleVolLineFlows, Time  = (
            inputs['index']['i'],
            inputs['index']['o'],
            inputs['i'],
            inputs['o'],
            inputs['p'],
            inputs['q'],
            inputs['CycleVolIn2'],
            inputs['CycleVolOut2'],
            inputs['Bounds'],
            inputs['CycleVolLineFlows'],
            inputs['Time']
        )

        #-------------------------------------------------------------------------------------------------------------              
        # Inflow Constraints
        #
        lst = list(set([t[0:4] for t in i_index]))            
        for tup in lst:
            tank = tup[0]; line = tup[1]; prod = tup[2]; j = tup[3]
            model.addConstr(i[tank, line, prod, j] - p[tank, line, prod, j] * 1000  <= 0)
            model.addConstr(i[tank, line, prod, j] - p[tank, line, prod, j] * 0.001  >= 0) 
            
        #-------------------------------------------------------------------------------------------------------------                        
        # Outflow constraints
        #
        lst = list(set([t[0:4] for t in o_index]))            
        for tup in lst:
            tank = tup[0]; line = tup[1]; prod = tup[2]; j = tup[3]
            model.addConstr(o[tank, line, prod, j] - q[tank, line, prod, j] * 1000 <= 0)
            model.addConstr(o[tank, line, prod, j] - q[tank, line, prod, j] * 0.001 >= 0) 
        
        #-------------------------------------------------------------------------------------------------------------              
        # Inflow Constraints
        #
        df = CycleVolLineFlows
        unique_combinations = df[['Line', 'Product']].drop_duplicates()

        for index, row in unique_combinations.iterrows():

            line = row['Line']
            prod = row['Product']
            if line in ['01', '02']:
                for time in Time:
                    exists = (df['Line'] == line) & (df['Product'] == prod) & (df['Time'] == time)
                    if exists.any():
                        filtered_data = df[(df['Time'] == time) & (df['Line'] == line) & (df['Product'] == prod)]
                        vol = filtered_data['Hourly_Vol'].iloc[0]
                        model.addConstr(p.sum('*', line, prod, time) == 1)      
                        model.addConstr(i.sum('*', line, prod, time) == vol)
                    else:
                        model.addConstr(p.sum('*', line, prod, time) == 0)      
                        model.addConstr(i.sum('*', line, prod, time) == 0)
            else:
                for time in Time:
                    exists = (df['Line'] == line) & (df['Product'] == prod) & (df['Time'] == time)
                    if exists.any():
                        filtered_data = df[(df['Time'] == time) & (df['Line'] == line) & (df['Product'] == prod)]
                        vol = filtered_data['Hourly_Vol'].iloc[0]
                        model.addConstr(q.sum('*', line, prod, time) == 1)      
                        model.addConstr(o.sum('*', line, prod, time) == vol)
                    else:
                        model.addConstr(q.sum('*', line, prod, time) == 0)      
                        model.addConstr(o.sum('*', line, prod, time) == 0)
        
        return (model)

    @staticmethod
    def model_timeline(ID, model, inputs):
        
        """
        Applies constraints to the given mathematical optimization `model` based on specific time constraints from the provided `inputs`.

        This function restricts the values of production (`p`) and consumption (`q`) variables based on time constraints
        defined in an external CSV file. It uses the timeline data from the CSV file to limit the values of `p` and `q`
        for specific tanks, lines, products, and time steps.

        Parameters:
            ID (int): An identifier or key used to locate the corresponding timeline CSV file.
            model (OptimizationModel): The mathematical optimization model to which constraints are added.
            inputs (dict): A dictionary containing input data required to create constraints.
                The dictionary should have the following keys:
                    - 'index': A dictionary containing information about 'i' and 'o' indexes.
                        It should have the following keys:
                            - 'i': A list of tuples representing the 'i' index with format (tank, line, prod, time, ...)
                            - 'o': A list of tuples representing the 'o' index with format (tank, line, prod, time, ...)
                    - 'p': The optimization variable 'p' representing production in the model.
                    - 'i': The optimization variable 'i' representing the inflow of tanks in the model.
                    - 'q': The optimization variable 'q' representing consumption in the model.

        Returns:
            OptimizationModel: The `model` with added constraints based on the timeline data.

        Example:
            # Assume 'model' and 'inputs' are defined.
            ID = 123  # The identifier for the corresponding timeline CSV file.
            model = function_model_const9(ID, model, inputs)

        Note:
            - The CSV file containing timeline data should be named 'opt_timeline_<ID>.csv', where <ID> is the given `ID`.
            - The timeline data should include information about lines, products, and their time constraints.
            - The function will read the timeline data from the CSV file and apply constraints on `p` and `q` variables
            based on the time limits specified in the timeline data.
            - If the time step of a variable falls outside the specified time limits for a given line and product,
            the function sets the corresponding variable to zero in the optimization model, effectively restricting
            the production or consumption at that specific time step and location.
            - The function ensures that production and consumption occur within the valid time ranges specified in the timeline data.
        """
        
        i_index, o_index, p, i, o, q = (
            inputs['index']['i'],
            inputs['index']['o'],
            inputs['p'],
            inputs['i'],
            inputs['o'],
            inputs['q']
        )    

        opt_1_timeline = pd.read_csv ('results/opt_timeline_' + str(ID) + '.csv')
        def line(df):
            if (df['Line'] in [1]):
                return '01'
            elif (df['Line'] in [2]):
                return '02'
            else:
                return str(df['Line'])
        opt_1_timeline['Line'] = opt_1_timeline.apply(line, axis = 1)

        lst = list(set([t[0:4] for t in i_index])) 
        for tup in lst:
            tank = tup[0]; line = tup[1]; prod = tup[2]; time = tup[3]
            a = opt_1_timeline[(opt_1_timeline['Line'] == line) & (opt_1_timeline['Product'] == prod)]   
            if (len(a.index) > 0):
                if ((time < a.iloc[0]["Time_min_f"]) | (time > a.iloc[0]["Time_max_f"])):
                    model.addConstr(p[tank, line, prod, time] == 0)
                    model.addConstr(i[tank, line, prod, time] == 0)

        lst = list(set([t[0:4] for t in o_index])) 
        for tup in lst:
            tank = tup[0]; line = tup[1]; prod = tup[2]; time = tup[3]
            a = opt_1_timeline[(opt_1_timeline['Line'] == line) & (opt_1_timeline['Product'] == prod)]   
            if (len(a.index) > 0):
                if ((time < a.iloc[0]["Time_min_f"]) | (time > a.iloc[0]["Time_max_f"])):
                    model.addConstr(q[tank, line, prod, time] == 0)
                    model.addConstr(o[tank, line, prod, time] == 0)
                            
        return (model)  
    
    @staticmethod
    def model_sequencing(model, inputs):

        """
        Adds specific constraints to the given mathematical optimization `model` based on the provided `inputs`.

        This function adds constraints to the optimization model for tank-to-line operations (`mo`) and line-to-tank
        operations (`mi`). It restricts the simultaneous flow between certain tanks and lines to ensure consistency
        and meet the constraints of the system.

        Parameters:
            model (OptimizationModel): The mathematical optimization model to which constraints are added.
            inputs (dict): A dictionary containing input data required to create constraints.
                The dictionary should have the following keys:
                    - 'index': A dictionary containing information about 'i', 'o', and 'o' indexes.
                        It should have the following keys:
                            - 'i': A list of tuples representing the 'i' index with format (tank, line, prod, time, ...)
                            - 'o': A list of tuples representing the 'o' index with format (tank, line, prod, time, ...)
                    - 'mo': The optimization variable 'mo' representing tank-to-line operations in the model.
                    - 'mi': The optimization variable 'mi' representing line-to-tank operations in the model.
                    - 'p': The optimization variable 'p' representing production in the model.
                    - 'q': The optimization variable 'q' representing consumption in the model.
                    - 'T': The total number of time steps in the model.

        Returns:
            OptimizationModel: The `model` with added constraints.

        Example:
            # Assume 'model' and 'inputs' are defined.
            model = function_model_const6(model, inputs)

        Note:
            - The 'mo' and 'mi' optimization variables must be previously defined in the model before calling this function.
            - The function will create constraints for tank-to-line operations (`mo`) based on the provided 'o' index and time-dependent information.
            - The function will create constraints for line-to-tank operations (`mi`) based on the provided 'i' index and time-dependent information.
            - The constraints ensure that simultaneous flow between certain tanks and lines follows specific rules.
            - The function applies different constraints for specific lines (e.g., '01', '02', '13', '14', etc.) and products (e.g., 'A', 'D', '54').
            - The constraints aim to ensure the consistency of the system and meet the specified capacity constraints.
        """

        i_index, o_index, mo, mi, p, q, T = (
            inputs['index']['i'],
            inputs['index']['o'],
            inputs['mo'],
            inputs['mi'],
            inputs['p'],
            inputs['q'],
            inputs['T']
        )

        #-------------------------------------------------------------------------------------------------------------
        #Constraint 9: Marker
        # 
        lst = [tup[0:3] for tup in o_index if tup[3] == 0]
        for tpl in lst:
            tank = tpl[0]; line = tpl[1]; prod = tpl[2] 
            for j in list(range(T)):
                for k in list(range(j + 1)):
                    model.addConstr(mo[line, prod, j] >= q[tank, line, prod, k])

        lst = [tup[0:3] for tup in i_index if tup[3] == 0]
        for tpl in lst:
            tank = tpl[0]; line = tpl[1]; prod = tpl[2] 
            for j in list(range(T)):
                for k in list(range(j + 1)):
                    model.addConstr(mi[line, prod, j] >= p[tank, line, prod, k])   

        #-------------------------------------------------------------------------------------------------------------
        # Line - 01
        #      
        line = '01'
        #
        lst1 = [tup[0:3] for tup in i_index if tup[1] == line and tup[3] == 0 and tup[2] == 'A']
        lst2 = [tup[1:3] for tup in i_index if tup[1] == line and tup[3] == 0 and tup[2] not in ['A']]
        lst2 = list(set(lst2))
        for tpl1 in lst1:
            tank1 = tpl1[0]
            line1 = tpl1[1]
            prod1 = tpl1[2]
            for tpl2 in lst2:
                line2 = tpl2[0]
                prod2 = tpl2[1]
                for j in list(range(T)):
                    model.addConstr(p[tank1, line1, prod1, j] + mi[line2, prod2, j] <= 1) 

        #-------------------------------------------------------------------------------------------------------------
        # Line - 02
        #      
        line = '02'
        #
        lst1 = [tup[0:3] for tup in i_index if tup[1] == line and tup[3] == 0 and tup[2] == '54']
        lst2 = [tup[1:3] for tup in i_index if tup[1] == line and tup[3] == 0 and tup[2] not in ['54']]
        lst2 = list(set(lst2))
        for tpl1 in lst1:
            tank1 = tpl1[0]
            line1 = tpl1[1]
            prod1 = tpl1[2]
            for tpl2 in lst2:
                line2 = tpl2[0]
                prod2 = tpl2[1]
                for j in list(range(T)):
                    model.addConstr(p[tank1, line1, prod1, j] + mi[line2, prod2, j] <= 1)             

        #-------------------------------------------------------------------------------------------------------------
        # Line - 13
        #
        line = '13'
        prds = ['A', 'D', '54']
        
        for i in range(len(prds)):

            lst1 = [tup[0:3] for tup in o_index if tup[1] == line and tup[3] == 0 and tup[2] == prds[i]]
            lst2 = [tup[1:3] for tup in o_index if tup[1] == line and tup[3] == 0 and tup[2] not in prds[0:(i+1)]]
            lst2 = list(set(lst2))
            for tpl1 in lst1:
                tank1 = tpl1[0]
                line1 = tpl1[1]
                prod1 = tpl1[2]
                for tpl2 in lst2:
                    line2 = tpl2[0]
                    prod2 = tpl2[1]
                    for j in list(range(T)):
                        model.addConstr(q[tank1, line1, prod1, j] + mo[line2, prod2, j] <= 1)    


        #-------------------------------------------------------------------------------------------------------------
        # Line - 14
        #
        line = '14'
        prds = ['A', 'D', '54']
        
        for i in range(len(prds)):

            lst1 = [tup[0:3] for tup in o_index if tup[1] == line and tup[3] == 0 and tup[2] == prds[i]]
            lst2 = [tup[1:3] for tup in o_index if tup[1] == line and tup[3] == 0 and tup[2] not in prds[0:(i+1)]]
            lst2 = list(set(lst2))
            for tpl1 in lst1:
                tank1 = tpl1[0]
                line1 = tpl1[1]
                prod1 = tpl1[2]
                for tpl2 in lst2:
                    line2 = tpl2[0]
                    prod2 = tpl2[1]
                    for j in list(range(T)):
                        model.addConstr(q[tank1, line1, prod1, j] + mo[line2, prod2, j] <= 1)        

            

        #-------------------------------------------------------------------------------------------------------------                       
        # Line - 15
        #
        line = '15'
        prds = ['A', 'D', '54']
        
        for i in range(len(prds)):

            lst1 = [tup[0:3] for tup in o_index if tup[1] == line and tup[3] == 0 and tup[2] == prds[i]]
            lst2 = [tup[1:3] for tup in o_index if tup[1] == line and tup[3] == 0 and tup[2] not in prds[0:(i+1)]]
            lst2 = list(set(lst2))
            for tpl1 in lst1:
                tank1 = tpl1[0]
                line1 = tpl1[1]
                prod1 = tpl1[2]
                for tpl2 in lst2:
                    line2 = tpl2[0]
                    prod2 = tpl2[1]
                    for j in list(range(T)):
                        model.addConstr(q[tank1, line1, prod1, j] + mo[line2, prod2, j] <= 1) 

        #-------------------------------------------------------------------------------------------------------------                   
        # Line - 16
        #
        line = '16'
        prds = ['A', 'D', '54']
        
        for i in range(len(prds)):

            lst1 = [tup[0:3] for tup in o_index if tup[1] == line and tup[3] == 0 and tup[2] == prds[i]]
            lst2 = [tup[1:3] for tup in o_index if tup[1] == line and tup[3] == 0 and tup[2] not in prds[0:(i+1)]]
            lst2 = list(set(lst2))
            for tpl1 in lst1:
                tank1 = tpl1[0]
                line1 = tpl1[1]
                prod1 = tpl1[2]
                for tpl2 in lst2:
                    line2 = tpl2[0]
                    prod2 = tpl2[1]
                    for j in list(range(T)):
                        model.addConstr(q[tank1, line1, prod1, j] + mo[line2, prod2, j] <= 1)          

        #-------------------------------------------------------------------------------------------------------------                        
        # Line - 17
        #
        line = '17'
        prds = ['A', 'D', '54']
        
        for i in range(len(prds)):

            lst1 = [tup[0:3] for tup in o_index if tup[1] == line and tup[3] == 0 and tup[2] == prds[i]]
            lst2 = [tup[1:3] for tup in o_index if tup[1] == line and tup[3] == 0 and tup[2] not in prds[0:(i+1)]]
            lst2 = list(set(lst2))
            for tpl1 in lst1:
                tank1 = tpl1[0]
                line1 = tpl1[1]
                prod1 = tpl1[2]
                for tpl2 in lst2:
                    line2 = tpl2[0]
                    prod2 = tpl2[1]
                    for j in list(range(T)):
                        model.addConstr(q[tank1, line1, prod1, j] + mo[line2, prod2, j] <= 1)            

        #-------------------------------------------------------------------------------------------------------------                        
        # Line - 18
        #
        line = '18'
        prds = ['A', 'D', '54']
        
        for i in range(len(prds)):

            lst1 = [tup[0:3] for tup in o_index if tup[1] == line and tup[3] == 0 and tup[2] == prds[i]]
            lst2 = [tup[1:3] for tup in o_index if tup[1] == line and tup[3] == 0 and tup[2] not in prds[0:(i+1)]]
            lst2 = list(set(lst2))
            for tpl1 in lst1:
                tank1 = tpl1[0]
                line1 = tpl1[1]
                prod1 = tpl1[2]
                for tpl2 in lst2:
                    line2 = tpl2[0]
                    prod2 = tpl2[1]
                    for j in list(range(T)):
                        model.addConstr(q[tank1, line1, prod1, j] + mo[line2, prod2, j] <= 1)  

        #-------------------------------------------------------------------------------------------------------------                        
        # Line - 19
        #
        line = '19'
        prds = ['A', 'D', '54']
        
        for i in range(len(prds)):

            lst1 = [tup[0:3] for tup in o_index if tup[1] == line and tup[3] == 0 and tup[2] == prds[i]]
            lst2 = [tup[1:3] for tup in o_index if tup[1] == line and tup[3] == 0 and tup[2] not in prds[0:(i+1)]]
            lst2 = list(set(lst2))
            for tpl1 in lst1:
                tank1 = tpl1[0]
                line1 = tpl1[1]
                prod1 = tpl1[2]
                for tpl2 in lst2:
                    line2 = tpl2[0]
                    prod2 = tpl2[1]
                    for j in list(range(T)):
                        model.addConstr(q[tank1, line1, prod1, j] + mo[line2, prod2, j] <= 1)  
                    
        #-------------------------------------------------------------------------------------------------------------                        
        # Line - 20
        #
        line = '20'
        prds = ['A', 'D', '54']
        
        for i in range(len(prds)):

            lst1 = [tup[0:3] for tup in o_index if tup[1] == line and tup[3] == 0 and tup[2] == prds[i]]
            lst2 = [tup[1:3] for tup in o_index if tup[1] == line and tup[3] == 0 and tup[2] not in prds[0:(i+1)]]
            lst2 = list(set(lst2))
            for tpl1 in lst1:
                tank1 = tpl1[0]
                line1 = tpl1[1]
                prod1 = tpl1[2]
                for tpl2 in lst2:
                    line2 = tpl2[0]
                    prod2 = tpl2[1]
                    for j in list(range(T)):
                        model.addConstr(q[tank1, line1, prod1, j] + mo[line2, prod2, j] <= 1)  
            
        #-------------------------------------------------------------------------------------------------------------                        
        # Line - 1A
        #
        line = '1A'
        #
        lst1 = [tup[0:3] for tup in o_index if tup[1] == line and tup[3] == 0 and tup[2] == 'A']
        lst2 = [tup[1:3] for tup in o_index if tup[1] == line and tup[3] == 0 and tup[2] not in ['A']]
        lst2 = list(set(lst2))
        for tpl1 in lst1:
            tank1 = tpl1[0]
            line1 = tpl1[1]
            prod1 = tpl1[2]
            for tpl2 in lst2:
                line2 = tpl2[0]
                prod2 = tpl2[1]
                for j in list(range(T)):
                    model.addConstr(q[tank1, line1, prod1, j] + mo[line2, prod2, j] <= 1) 
                    
        #-------------------------------------------------------------------------------------------------------------                   
        # Line - 2A
        #
        line = '2A'
        #
        lst1 = [tup[0:3] for tup in o_index if tup[1] == line and tup[3] == 0 and tup[2] == '54']
        lst2 = [tup[1:3] for tup in o_index if tup[1] == line and tup[3] == 0 and tup[2] not in ['54']]
        lst2 = list(set(lst2))
        for tpl1 in lst1:
            tank1 = tpl1[0]
            line1 = tpl1[1]
            prod1 = tpl1[2]
            for tpl2 in lst2:
                line2 = tpl2[0]
                prod2 = tpl2[1]
                for j in list(range(T)):
                    model.addConstr(q[tank1, line1, prod1, j] + mo[line2, prod2, j] <= 1)             
                    

        return (model)
    
    @staticmethod
    def model_flow_const1(model, inputs):

        """
        Adds constraints to the given mathematical optimization `model` based on the provided `inputs`.

        This function adds constraints to the optimization model for production (`p`) and consumption (`q`) operations
        based on the provided index data and capacity constraints. It also applies absolute difference constraints
        to certain variables to limit their fluctuations.

        Parameters:
            model (OptimizationModel): The mathematical optimization model to which constraints are added.
            inputs (dict): A dictionary containing input data required to create constraints.
                The dictionary should have the following keys:
                    - 'Capacity': A dictionary containing capacity constraints for specific production lines and products.
                    - 'index': A dictionary containing information about 'li', 'lo', and 'o' indexes.
                        It should have the following keys:
                            - 'li': A list of tuples representing the 'li' index with format (line, prod, time, ...)
                            - 'lo': A list of tuples representing the 'lo' index with format (line, prod, time, ...)
                            - 'o': A list of tuples representing the 'o' index with format (line, prod, time, ...)
                    - 'tlpo': A dictionary containing time-dependent information for production and consumption.
                    - 'li': The optimization variable 'li' representing production in the model.
                    - 'lo': The optimization variable 'lo' representing consumption in the model.
                    - 'abs_li': Auxiliary optimization variable 'abs_li' used for absolute difference constraints.
                    - 'abs_lo': Auxiliary optimization variable 'abs_lo' used for absolute difference constraints.
                    - 'p': The optimization variable 'p' representing production in the model.
                    - 'q': The optimization variable 'q' representing consumption in the model.
                    - 'Time': A list of time steps in the model.
                    - 'aux_li': Auxiliary optimization variable 'aux_li' used for intermediate calculations.
                    - 'aux_lo': Auxiliary optimization variable 'aux_lo' used for intermediate calculations.
                    - 'T': The total number of time steps in the model.

        Returns:
            OptimizationModel: The `model` with added constraints.

        Example:
            # Assume 'model' and 'inputs' are defined.
            model = function_model_const5(model, inputs)

        Note:
            - The 'li', 'lo', 'p', and 'q' optimization variables must be previously defined in the model before calling this function.
            - The function will create constraints for production (`p`) based on the provided 'li' index and time-dependent information.
            - The function will create constraints for consumption (`q`) based on the provided 'lo' index and time-dependent information.
            - The function will apply absolute difference constraints to certain variables (auxiliary variables) to limit their fluctuations.
            - The constraints ensure that production and consumption operations follow the specified capacity constraints.
            - The absolute difference constraints limit the changes in production and consumption between consecutive time steps to avoid abrupt variations.
        """
        
        Capacity, li_index, lo_index, o_index, tlpo, li, lo, abs_li, abs_lo, p, q, Time, aux_li, aux_lo, T = (
            inputs['Capacity'],
            inputs['index']['li'],
            inputs['index']['lo'],
            inputs['index']['o'],
            inputs['tlpo'],
            inputs['li'],
            inputs['lo'],
            inputs['abs_li'],
            inputs['abs_lo'],
            inputs['p'],
            inputs['q'],
            inputs['Time'],
            inputs['aux_li'],
            inputs['aux_lo'],
            inputs['T']
        )
        
        #-------------------------------------------------------------------------------------------------------------                    
        # 
        #  
        lst = set([tup[0:2] for tup in li_index])
        for tup in lst:
            line = tup[0]; prod = tup[1]
            for j in Time:
                model.addConstr(p.sum('*', line, prod, j) == li[line, prod, j]) 

        lst = set([tup[0:2] for tup in li_index])
        for tup in lst:
            line = tup[0]; prod = tup[1]
            model.addConstr(aux_li[line, prod, 0] == 0)
            model.addConstr(abs_li[line, prod, 0] == 0)
            for j in list(range(1, T)):
                model.addConstr(aux_li[line, prod, j] == li[line, prod, j] - li[line, prod, j - 1])
                model.addGenConstrAbs(abs_li[line, prod, j], aux_li[line, prod, j], "absConstr_li")

        for tup in lst:
            line = tup[0]; prod = tup[1];        
            model.addConstr(abs_li.sum(line, prod, '*') <= 20)

        #-------------------------------------------------------------------------------------------------------------                    
        # 
        #                 
        lst = set([tup[0:2] for tup in lo_index])
        for tup in lst:
            line = tup[0]; prod = tup[1]
            for j in Time:
                model.addConstr(q.sum('*', line, prod, j) == lo[line, prod, j])                 

        lst = set([tup[0:2] for tup in lo_index])
        for tup in lst:
            line = tup[0]; prod = tup[1]
            model.addConstr(aux_lo[line, prod, 0] == 0)
            model.addConstr(abs_lo[line, prod, 0] == 0)
            for j in list(range(1, T)):
                model.addConstr(aux_lo[line, prod, j] == lo[line, prod, j] - lo[line, prod, j - 1])
                model.addGenConstrAbs(abs_lo[line, prod, j], aux_lo[line, prod, j], "absConstr_lo")

        for tup in lst:
            line = tup[0]; prod = tup[1];        
            model.addConstr(abs_lo.sum(line, prod, '*') <= 20) 
        
        return (model)

    @staticmethod
    def model_flow_const2(model, inputs):
            
        """
        Adds specific constraints to the given mathematical optimization `model` based on the provided `inputs`.

        This function adds constraints to the optimization model for tank inflow (`aux_ti`), tank outflow (`aux_to`), and
        their absolute values (`abs_ti` and `abs_to`). The constraints ensure that the inflow and outflow rates for tanks
        and lines follow specific rules and meet the specified capacity constraints.

        Parameters:
            model (OptimizationModel): The mathematical optimization model to which constraints are added.
            inputs (dict): A dictionary containing input data required to create constraints.
                The dictionary should have the following keys:
                    - 'index': A dictionary containing information about 'i', 'o', 'ti', 'li', 'lo', and 'to' indexes.
                        It should have the following keys:
                            - 'i': A list of tuples representing the 'i' index with format (tank, line, prod, time, ...)
                            - 'o': A list of tuples representing the 'o' index with format (tank, line, prod, time, ...)
                            - 'ti': A list of tuples representing the 'ti' index with format (tank, line, prod, time, ...)
                            - 'li': A list of tuples representing the 'li' index with format (tank, line, prod, time, ...)
                            - 'lo': A list of tuples representing the 'lo' index with format (tank, line, prod, time, ...)
                            - 'to': A list of tuples representing the 'to' index with format (tank, line, prod, time, ...)
                    - 'aux_ti': The optimization variable 'aux_ti' representing tank inflow in the model.
                    - 'aux_to': The optimization variable 'aux_to' representing tank outflow in the model.
                    - 'abs_ti': The optimization variable 'abs_ti' representing the absolute value of tank inflow in the model.
                    - 'abs_to': The optimization variable 'abs_to' representing the absolute value of tank outflow in the model.
                    - 'p': The optimization variable 'p' representing production in the model.
                    - 'q': The optimization variable 'q' representing consumption in the model.
                    - 'T': The total number of time steps in the model.
                    - 'Time': A list of time steps in the model.

        Returns:
            OptimizationModel: The `model` with added constraints.

        Example:
            # Assume 'model' and 'inputs' are defined.
            model = function_model_const8(model, inputs)

        Note:
            - The 'aux_ti', 'aux_to', 'abs_ti', and 'abs_to' optimization variables must be previously defined in the model before calling this function.
            - The function will create constraints for tank inflow (`aux_ti`) based on the provided 'ti' index and time-dependent information.
            - The function will create constraints for tank outflow (`aux_to`) based on the provided 'to' index and time-dependent information.
            - The constraints ensure that the inflow and outflow rates for tanks and lines follow specific rules.
            - The function applies different constraints for specific tanks, lines, and products.
            - The constraints aim to ensure the consistency of the system and meet the specified capacity constraints.
        """    
            
            
        i_index, o_index, ti_index, li_index, lo_index, aux_ti, aux_li, aux_lo, aux_to, abs_ti, abs_to, abs_li, abs_lo, to_index, p, q, li, lo, tlpo, t, T, Time, flow_constraints, flow_constraints_univ_1, flow_constraints_univ_2 = (
            inputs['index']['i'],
            inputs['index']['o'],
            inputs['index']['ti'],
            inputs['index']['li'],
            inputs['index']['lo'],
            inputs['aux_ti'],
            inputs['aux_li'],
            inputs['aux_lo'],
            inputs['aux_to'],
            inputs['abs_ti'],
            inputs['abs_to'],
            inputs['abs_li'],
            inputs['abs_lo'],
            inputs['index']['to'],
            inputs['p'],
            inputs['q'],
            inputs['li'],
            inputs['lo'],
            inputs['tlpo'],
            inputs['t'], 
            inputs['T'],
            inputs['Time'],
            inputs['flow_constraints'],
            inputs['flow_constraints_univ_1'],
            inputs['flow_constraints_univ_2']
        )    
            
        #-------------------------------------------------------------------------------------------------------------                     
        #
        #
        lst = set([tup[0:3] for tup in ti_index])
        for tup in lst:
            tank = tup[0]; line = tup[1]; prod = tup[2]
            model.addConstr(aux_ti[tank, line, prod, 0] == 0)
            model.addConstr(abs_ti[tank, line, prod, 0] == 0)
            for j in list(range(1, T)):
                model.addConstr(aux_ti[tank, line, prod, j] == p[tank, line, prod, j] - p[tank, line, prod, j - 1])
                model.addGenConstrAbs(abs_ti[tank, line, prod, j], aux_ti[tank, line, prod, j], "absConstr_ti")
            
        # Apply the customized constraints           
        tanks = [item['Tank'] for item in flow_constraints]           
        for tup in lst:
            tank = tup[0]; line = tup[1]; prod = tup[2]
            if tank in tanks:
                value = next((item['Inbound'] for item in flow_constraints if item['Tank'] == tank), None)   
                model.addConstr(abs_ti.sum(tank, line, prod, '*') <= 2 * value)
            else:
                model.addConstr(abs_ti.sum(tank, line, prod, '*') <= 2 * flow_constraints_univ_1)
                

        #-------------------------------------------------------------------------------------------------------------                     
        # 
        # 
        lst = set([tup[0:3] for tup in to_index])
        for tup in lst:
            tank = tup[0]; line = tup[1]; prod = tup[2]
            model.addConstr(aux_to[tank, line, prod, 0] == 0)
            model.addConstr(abs_to[tank, line, prod, 0] == 0)
            for j in list(range(1, T)):
                model.addConstr(aux_to[tank, line, prod, j] == q[tank, line, prod, j] - q[tank, line, prod, j - 1])
                model.addGenConstrAbs(abs_to[tank, line, prod, j], aux_to[tank, line, prod, j], "absConstr_to")

        # Apply the customized constraints           
        tanks = [item['Tank'] for item in flow_constraints]           
        for tup in lst:
            tank = tup[0]; line = tup[1]; prod = tup[2]
            if tank in tanks:
                value = next((item['Outbound'] for item in flow_constraints if item['Tank'] == tank), None)   
                model.addConstr(abs_to.sum(tank, line, prod, '*') <= 2 * value)
            else:
                model.addConstr(abs_to.sum(tank, line, prod, '*') <= 2 * flow_constraints_univ_2)    
                
        return (model)

    @staticmethod
    def model_tank_const1(model, inputs):
        
        i_index, o_index, p, q, tk, tanks_to_use = (
            inputs['index']['i'],
            inputs['index']['o'],
            inputs['p'],
            inputs['q'],
            inputs['tk'],
            inputs['Tanks_to_use']
        )    
        
        lst = list(set([t[0:4] for t in i_index]))            
        for tup in lst:
            tank = tup[0]; line = tup[1]; prod = tup[2]; time = tup[3]
            if tank not in tanks_to_use: 
                model.addConstr(p[tank, line, prod, time] == 0)
        
        lst = list(set([t[0:4] for t in o_index]))            
        for tup in lst:
            tank = tup[0]; line = tup[1]; prod = tup[2]; time = tup[3]
            if tank not in tanks_to_use: 
                model.addConstr(q[tank, line, prod, time] == 0)

        
        return (model) 
    
    @staticmethod
    def model_tank_const2(model, inputs):
        
        """
        Applies constraints to the given mathematical optimization `model` to limit the consumption (`q`) based on
        the 'tlpo' (Tank Line Product Output) values.

        This function sets constraints on the consumption variables (`q`) to ensure that they do not exceed the available
        'tlpo' values for each tank, line, and product combination. It also sets an upper limit on the total 'tlpo' value
        for each line and product combination.

        Parameters:
            model (OptimizationModel): The mathematical optimization model to which constraints are added.
            inputs (dict): A dictionary containing input data required to create constraints.
                The dictionary should have the following keys:
                    - 'index': A dictionary containing information about 'i' and 'o' indexes.
                        It should have the following keys:
                            - 'i': A list of tuples representing the 'i' index with format (tank, line, prod, time, ...)
                            - 'o': A list of tuples representing the 'o' index with format (tank, line, prod, time, ...)
                    - 'p': The optimization variable 'p' representing production in the model.
                    - 'q': The optimization variable 'q' representing consumption in the model.
                    - 'tlpo': A dictionary representing the 'tlpo' (Tank Line Product Output) values.
                    It should map tank, line, product combinations to their respective output limits.

        Returns:
            OptimizationModel: The `model` with added constraints to limit the consumption variables (`q`) based on
            the available 'tlpo' values for each tank, line, and product combination.

        Example:
            # Assume 'model' and 'inputs' are defined.
            model = function_model_const11(model, inputs)

        Note:
            - This constraint ensures that a line can receive product from at most X number of tanks
            - at the same time.
        """
        
        i_index, o_index, p, q, tlpo = (
            inputs['index']['i'],
            inputs['index']['o'],
            inputs['p'],
            inputs['q'],
            inputs['tlpo']
        )    
        
        lst = set([tup[0:4] for tup in o_index])
        for tup in lst:
            tank = tup[0]; line = tup[1]; prod = tup[2]; j = tup[3]
            model.addConstr(q[tank, line, prod, j] <= tlpo[tank, line, prod])

        lst = set([tup[1:3] for tup in o_index])
        for tup in lst:
            line = tup[0]; prod = tup[1]       
            model.addConstr(tlpo.sum('*', line, prod) <= 4)  

        return (model)

    @staticmethod
    def model_tank_const3(model, inputs):
        
        """
        Applies constraints to the given mathematical optimization `model` to limit the production (`p`) and consumption (`q`)
        based on the available tank capacities (`tk`).

        This function sets constraints on the production variables (`p`) and consumption variables (`q`) to ensure that they
        do not exceed the capacity of the respective tanks. The capacity of each tank is represented by the 'tk' (Tank Capacity)
        values.

        Parameters:
            model (OptimizationModel): The mathematical optimization model to which constraints are added.
            inputs (dict): A dictionary containing input data required to create constraints.
                The dictionary should have the following keys:
                    - 'index': A dictionary containing information about 'i' and 'o' indexes.
                        It should have the following keys:
                            - 'i': A list of tuples representing the 'i' index with format (tank, line, prod, time, ...)
                            - 'o': A list of tuples representing the 'o' index with format (tank, line, prod, time, ...)
                    - 'p': The optimization variable 'p' representing production in the model.
                    - 'q': The optimization variable 'q' representing consumption in the model.
                    - 'tk': A dictionary representing the tank capacities.
                    It should map each tank to its respective capacity value.

        Returns:
            OptimizationModel: The `model` with added constraints to limit the production variables (`p`) and consumption
            variables (`q`) based on the available tank capacities (`tk`).

        Example:
            # Assume 'model' and 'inputs' are defined.
            model = function_model_const12(model, inputs)

        Note:
            - The function ensures that the production (`p`) and consumption (`q`) variables do not exceed the available tank
            capacities represented by the 'tk' values. If the production or consumption demand for a specific tank, line,
            product, and time combination is greater than the tank's capacity, the function sets the production and
            consumption values to the tank's capacity.
            - By applying these constraints, the function prevents overproduction and overconsumption, ensuring that the
            production and consumption align with the available tank capacities.
        """

        i_index, o_index, p, q, tk = (
            inputs['index']['i'],
            inputs['index']['o'],
            inputs['p'],
            inputs['q'],
            inputs['tk']
        )    
        
        lst = set([tup[0:4] for tup in i_index])
        for tup in lst:
            tank = tup[0]; line = tup[1]; prod = tup[2]; j = tup[3]
            model.addConstr(p[tank, line, prod, j] <= tk[tank])

        lst = set([tup[0:4] for tup in o_index])
        for tup in lst:
            tank = tup[0]; line = tup[1]; prod = tup[2]; j = tup[3]
            model.addConstr(q[tank, line, prod, j] <= tk[tank])

        #model.addConstr(tk.sum('*') >= 14)     
            
        return (model)

    @staticmethod
    def model_obj1(model, inputs):
    
        x_index, x, t = (
            inputs['index']['x'],
            inputs['x'],
            inputs['t']
        )    
        
        lst = list(set([(t[0],t[2]) for t in x_index]))                    
        for tup in lst:
            tank = tup[0]; time = tup[1];
            model.addConstr(x.sum(tank, '*', time) <= t[0])

        return (model) 
    
    @staticmethod
    def model_obj2(model, inputs):
    
        x_index, o_index, ti_index, x, p, q, tlpo, t, aux_ti, abs_ti, T = (
            inputs['index']['x'],
            inputs['index']['o'],
            inputs['index']['ti'],
            inputs['x'],
            inputs['p'],
            inputs['q'],
            inputs['tlpo'],
            inputs['t'],
            inputs['aux_ti'],
            inputs['abs_ti'],
            inputs['T']
        )    
        
        lst = set([tup[0:3] for tup in ti_index])    
        for tup in lst:
            tank = tup[0]; line = tup[1]; prod = tup[2]
            model.addConstr(aux_ti[tank, line, prod, 0] == 0)
            model.addConstr(abs_ti[tank, line, prod, 0] == 0)
            for j in list(range(1, T)):
                model.addConstr(aux_ti[tank, line, prod, j] == p[tank, line, prod, j] - p[tank, line, prod, j - 1])
                model.addGenConstrAbs(abs_ti[tank, line, prod, j], aux_ti[tank, line, prod, j], "absConstr_ti")

        for tup in lst:
            tank = tup[0]; line = tup[1]; prod = tup[2]       
            model.addConstr(abs_ti.sum(tank, line, prod, '*') <= t[0])
            
        return (model)

    @staticmethod
    def model_obj3(model, inputs):
        
        x_index, o_index, ti_index, x, p, q, tlpo, t, aux_ti, abs_ti, T = (
            inputs['index']['x'],
            inputs['index']['o'],
            inputs['index']['ti'],
            inputs['x'],
            inputs['p'],
            inputs['q'],
            inputs['tlpo'],
            inputs['t'],
            inputs['aux_ti'],
            inputs['abs_ti'],
            inputs['T']
        ) 
        
        lst = set([tup[1:3] for tup in o_index])
        for tup in lst:
            line = tup[0]; prod = tup[1]       
            model.addConstr(tlpo.sum('*', line, prod) <= t[0])  
        return (model)  
    
    @staticmethod
    def model_obj4(model, inputs):
            
        tk, t = (
            inputs['tk'],
            inputs['t']
        )     
            
        model.addConstr(tk.sum('*') <= t[0])  
        return (model)

    @staticmethod
    def model_obj5(model, inputs):
            
        li_index, lo_index, abs_ti, abs_to, t = (
            inputs['index']['li'],
            inputs['index']['lo'],
            inputs['abs_ti'],
            inputs['abs_to'],
            inputs['t']
        )         

    #     lst = set([tup[0:2] for tup in li_index])
    #     for tup in lst:
    #         line = tup[0]; prod = tup[1];        
    #         model.addConstr(abs_li.sum(line, prod, '*') <= t[0])

    #     lst = set([tup[0:2] for tup in lo_index])
    #     for tup in lst:
    #         line = tup[0]; prod = tup[1];        
    #         model.addConstr(abs_lo.sum(line, prod, '*') <= t[0])

        #model.addConstr(abs_li.sum('*', '*', '*') + abs_lo.sum('*', '*', '*') <= t[0])
        model.addConstr(abs_ti.sum('*', '*', '*') + abs_to.sum('*', '*', '*') <= t[0])
        
        return (model)

    @staticmethod
    def model_obj6(model, inputs):
            
        li_index, lo_index, abs_li, abs_lo, t = (
            inputs['index']['li'],
            inputs['index']['lo'],
            inputs['abs_li'],
            inputs['abs_lo'],
            inputs['t']
        )         

        lst = set([tup[0:2] for tup in li_index])
        for tup in lst:
            line = tup[0]; prod = tup[1];        
            model.addConstr(abs_li.sum(line, prod, '*') <= t[0])

        return (model)

    @staticmethod
    def model_obj(model, inputs):
            
        objective = (
            inputs['objective']
        )     
        
        if objective == "selection1":    
            model = OptimizationModel.model_obj4(model, inputs)
        elif objective == "selection2":
            model = OptimizationModel.model_obj5(model, inputs)
        elif objective == "selection3":
            model = OptimizationModel.model_obj6(model, inputs)    
            
        return (model) 
    
    @staticmethod
    def model_stage3(ID, inputs):

        index, CycleVolIn2, CycleVolOut2, CycleVolExist, Bounds, Capacity, Tanks, T, Time, Cycle, CycleStart = (
            inputs['index'],
            inputs['CycleVolIn2'],
            inputs['CycleVolOut2'],
            inputs['CycleVolExist'],
            inputs['Bounds'],
            inputs['Capacity'],
            inputs['Tanks'],
            inputs['T'],
            inputs['Time'],
            inputs['Cycle'],
            inputs['CycleStart']
        )
        
        ret = {}
        #-------------------------------------------------------------------------------------------------------------              
        #
        # 
        
        model  = gp.Model('ATJ2')
        
        x_index = index['x']
        x       = model.addVars(x_index,  lb = 0.0, ub = 200, vtype = GRB.CONTINUOUS)
        abs_p   = model.addVars(x_index, lb = 0.0, vtype = GRB.INTEGER)
        abs_q   = model.addVars(x_index, lb = 0.0, vtype = GRB.INTEGER)
        
        i_index = index['i']
        i       = model.addVars(i_index,  lb = 0.0, ub = 30,  vtype = GRB.CONTINUOUS)
        p       = model.addVars(i_index,  vtype = GRB.BINARY) 
        aux_p   = model.addVars(i_index, lb = -1, ub = 1, vtype = GRB.INTEGER)
        
        o_index = index['o']
        o       = model.addVars(o_index,  lb = 0.0, ub = 10,  vtype = GRB.CONTINUOUS) 
        q       = model.addVars(o_index,  vtype = GRB.BINARY)
        aux_q   = model.addVars(o_index, lb = -1, ub = 1, vtype = GRB.INTEGER)

        mo_index = index['mo']
        mo       = model.addVars(mo_index, vtype = GRB.BINARY)
        
        mi_index = index['mi']
        mi       = model.addVars(mi_index, vtype = GRB.BINARY)
        
        li_index = index['li']
        li       = model.addVars(li_index, vtype = GRB.INTEGER)
        aux_li   = model.addVars(li_index, lb = -1, ub = 1, vtype = GRB.INTEGER)
        abs_li   = model.addVars(li_index, vtype = GRB.INTEGER)
        
        lo_index = index['lo']
        lo       = model.addVars(lo_index, vtype = GRB.INTEGER)
        aux_lo   = model.addVars(lo_index, lb = -1, ub = 1, vtype = GRB.INTEGER)
        abs_lo   = model.addVars(lo_index, vtype = GRB.INTEGER)
        
        to_index = index['to']    
        to       = model.addVars(to_index, vtype = GRB.INTEGER)
        aux_to   = model.addVars(to_index, lb = -1, ub = 1, vtype = GRB.INTEGER)
        abs_to   = model.addVars(to_index, vtype = GRB.INTEGER)

        ti_index = index['ti']
        ti       = model.addVars(ti_index, vtype = GRB.INTEGER)
        aux_ti   = model.addVars(ti_index, lb = -1, ub = 1, vtype = GRB.INTEGER)
        abs_ti   = model.addVars(ti_index, vtype = GRB.INTEGER)
        
        tk_index = index['tank']
        tk       = model.addVars(tk_index, vtype = GRB.BINARY)

        tlpo_index = index['tlpo']
        tlpo       = model.addVars(tlpo_index, vtype = GRB.BINARY)

        t          = model.addVars(1, lb = 0.0, vtype = GRB.CONTINUOUS)
    
        #-------------------------------------------------------------------------------------------------------------              
        #
        # 
        inputs.update({
            'x': x, 'abs_p': abs_p, 'abs_q': abs_q,
            'i': i, 'p': p, 'aux_p': aux_p,
            'o': o, 'q': q, 'aux_q': aux_q,
            'mo': mo, 'mi': mi,
            'lo': lo, 'aux_lo': aux_lo, 'abs_lo': abs_lo,
            'li': li, 'aux_li': aux_li, 'abs_li': abs_li,
            'to': to, 'aux_to': aux_to, 'abs_to': abs_to,
            'ti': ti, 'aux_ti': aux_ti, 'abs_ti': abs_ti,
            'tk': tk, 'tlpo': tlpo, 't': t
        })

                
        #-------------------------------------------------------------------------------------------------------------
        # Constraint 1: All p and q should sum to at most 1 across Tanks, Lines 
        # 1.1
        model = OptimizationModel.model_baseline_const1(model,  inputs)
        model = OptimizationModel.model_baseline_const2(model,  inputs)    
        model = OptimizationModel.model_baseline_const3_v2(model,  inputs)
        model = OptimizationModel.model_flow_const1(model,  inputs)
        model = OptimizationModel.model_flow_const2(model,  inputs)
        model = OptimizationModel.model_tank_const1(model, inputs)
        model = OptimizationModel.model_tank_const2(model, inputs)
        model = OptimizationModel.model_tank_const3(model, inputs)
        
        model = OptimizationModel.model_obj(model, inputs)
        
        #-------------------------------------------------------------------------------------------------------------                      
        # 
        # 
        model.setObjective(t[0], GRB.MINIMIZE)
        model.write("model.mps")
        model.Params.timelimit = 12 * 60
        model.optimize()
        
        #-------------------------------------------------------------------------------------------------------------                      
        # 
        # 
        status = model.Status
        if status == gp.GRB.OPTIMAL:
            ret_status = "Optimal solution found!"
        elif status == gp.GRB.INFEASIBLE:
            ret_status = "Model is infeasible."
        elif status == gp.GRB.LOADED:
            ret_status = "Model is loaded, but no solution information is available."
        elif status == gp.GRB.INF_OR_UNBD:
            ret_status = "Model is either infeasible or unbounded."    
        elif status == gp.GRB.UNBOUNDED:
            ret_status = "Model is unbounded."    
        elif status == gp.GRB.CUTOFF:
            ret_status = "Optimal objective for model is worse than the value specified in the Cutoff parameter. No solution information is available."
        elif status == gp.GRB.ITERATION_LIMIT:
            ret_status = "Optimization terminated because the total number of simplex iterations performed exceeded the value specified in the IterLimit parameter."
        elif status == gp.GRB.NODE_LIMIT:
            ret_status = "Optimization terminated because the total number of branch-and-cut nodes explored exceeded the value specified in the NodeLimit parameter."
        elif status == gp.GRB.TIME_LIMIT:
            ret_status = "Optimization terminated because the time expended exceeded the value specified in the TimeLimit parameter."    
        elif status == gp.GRB.SOLUTION_LIMIT:
            ret_status = "Optimization terminated because the number of solutions found reached the value specified in the SolutionLimit parameter." 
        elif status == gp.GRB.INTERRUPTED:
            ret_status = "Optimization is terminated by the user."
        elif status == gp.GRB.NUMERIC:
            ret_status = "Optimization was terminated due to unrecoverable numerical difficulties."
        elif status == gp.GRB.SUBOPTIMAL:
            ret_status = "Unable to satisfy optimality tolerances; a sub-optimal solution is available."    
            
            
        #-------------------------------------------------------------------------------------------------------------                      
        # 
        #
        if status == gp.GRB.OPTIMAL:
            DataAnalysis.store_variables(ID, inputs)
            schedule           = DataAnalysis.schedule(ID, inputs)
            analysis_inputFlow = DataAnalysis.inputFlow(Cycle)
            analysis_lines     = DataAnalysis.process_line_volumes()
            analysis_tanks     = DataAnalysis.process_tank_volumes()
        else: 
            schedule           = None
            analysis_inputFlow = None
            analysis_tanks     = None
            analysis_lines     = None
        
        ret ={}
        ret['Status']             = ret_status
        ret['Schedule']           = schedule
        ret['analysis_inputFlow'] = analysis_inputFlow
        ret['analysis_tanks']     = analysis_tanks
        ret['analysis_lines']     = analysis_lines
        
        return(ret)

