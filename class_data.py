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

class DataLocation:
    def __init__(self, name):
        self.Name = name
        self.Capacity = {}
        self.Tanks = []
        self.Tanks_073 = []
        self.Tanks_085 = []
        self.Bounds = {}
        self.topo_i = {}
        self.topo_o = {}

        self.Tanks, self.Tanks_073, self.Tanks_085, self.Capacity = self.load_tanks('input_location/capacities_ATJ.csv')
        self.Bounds = self.load_bounds('input_location/bounds_ATJ.csv')
        self.topo_i = self.load_topology('input_location/topoI_ATJ.csv')
        self.topo_o = self.load_topology('input_location/topoO_ATJ.csv')

        self.validate_topologies(self.topo_i, self.topo_o)
        self.validate_tanks(self.topo_i, self.Tanks)
        self.validate_lines(self.topo_i, self.topo_o, self.Bounds)

    def load_bounds(self, filepath):
        try:
            # Read the CSV file into a DataFrame
            df_read = pd.read_csv(filepath, encoding="ISO-8859-1", dtype={'Line': str, 'Lower': float, 'Upper': float})
            
            # Remove quotes from 'Line' if they exist and ensure it's a string
            df_read['Line'] = df_read['Line'].str.strip('"')
            
            # Convert the DataFrame to a dictionary with 'Line' as keys and {'l': Lower, 'u': Upper} as values
            bounds = df_read.set_index('Line').apply(lambda row: {'l': row['Lower'], 'u': row['Upper']}, axis=1).to_dict()

            return bounds

        except Exception as e:
            print(f"An error occurred in load_bounds: {e}")
            return {}
    
    def load_topology(self, filepath):
        try:
            # Read the CSV file into a DataFrame
            df_read = pd.read_csv(filepath, dtype={'Tank': int, 'Line': str, 'Product': str})
            topology = {}

            # Iterate through the DataFrame and build the topology dictionary
            for _, row in df_read.iterrows():
                tank = row['Tank']
                line = row['Line']
                product = row['Product']

                # Initialize the tank dict if not already present
                if tank not in topology:
                    topology[tank] = {}

                # Add the line and product with an initial value of 0
                topology[tank][line] = {product: 0}

            return topology
        
        except Exception as e:
            print(f"An error occurred in load_topology: {e}")
            return {}
    
    def load_tanks(self, filepath):
        try:
            # Read the CSV file
            df_read = pd.read_csv(filepath, encoding="ISO-8859-1")
            
            # Round the 'Working' column and convert to integer
            df_read['Working'] = (1000 * df_read['Working']).round().astype(int) // 1000
            
            # Create the 'Capacity' dictionary
            capacities = pd.Series(df_read['Working'].values + 10, index=df_read['Tank']).to_dict()
            
            # Create the 'Tanks' list
            tanks = df_read['Tank'].tolist()
            
            # Create the 'Tanks_073' and 'Tanks_085' lists
            tanks_073 = df_read[df_read['Classification'] == "Gas"]['Tank'].tolist()
            tanks_085 = df_read[df_read['Classification'] == "Oil"]['Tank'].tolist()
            
            return tanks, tanks_073, tanks_085, capacities

        except Exception as e:
            print(f"An error occurred in load_tanks: {e}")
            return [], [], [], {}

    def validate_topologies(self, topo_i, topo_o):
        if not list(topo_i.keys()) == list(topo_o.keys()):
            raise ValueError("Error: Input and Output topology keys are not the same.")

    def validate_tanks(self, topo_i, Tanks):
        if not list(topo_i.keys()) == sorted(Tanks):
            raise ValueError("Error: Topology and Tank keys do not match.")
            
    def validate_lines(self, topo_i, topo_o, Bounds):
        keys_o = set()
        for k, v in topo_o.items():
            keys_o.update(v.keys())

        keys_i = set()
        for k, v in topo_i.items():
            keys_i.update(v.keys())
            
        keys = list(keys_i) + list(keys_o)   
        if not sorted(keys) == sorted(list(Bounds.keys())):
            raise ValueError("Error: The topology and bound keys do not match.")
        

    def print_details(self):
            print("\n------- Tanks -------")
            print(self.Tanks)
            print("\n------- Tanks 073 -------")
            print(self.Tanks_073)
            print("\n------- Tanks 085 -------")
            print(self.Tanks_085)
            print("\n------- Capacities -------")
            print(self.Capacity)
            print("\n------- Bounds -------")
            print(self.Bounds)
            print("\n------- Topology I -------")
            print(self.topo_i)


class DataCycleLoader():

    def __init__(self):

        self.VolIn = self.load_volume('input_cycle/VolIn_v2.csv')
        self.VolOut = self.load_volume('input_cycle/VolOut_v2.csv')
        self.VolExist = self.load_volume_tank('input_cycle/VolExist_v2.csv')
        self.VolLineFlows = pd.read_csv('input_cycle/VolLineFlows.csv')   

    def load_volume(self, filepath):

        df_read = pd.read_csv(filepath, dtype={'Cycle': str, 'Line': str, 'Product': str, 'Volume': float})

        # Create an empty dictionary to store the cycle data.
        cycle_data = {}

        # Iterate over the DataFrame rows.
        for index, row in df_read.iterrows():
            cycle, line, product, volume = row['Cycle'], row['Line'], row['Product'], row['Volume']
            
            # Ensure the keys are strings to match the target dictionary structure.
            cycle, line = str(cycle), str(line)

            # Initialize nested dictionaries if not already present.
            if cycle not in cycle_data:
                cycle_data[cycle] = {}
            if line not in cycle_data[cycle]:
                cycle_data[cycle][line] = {}

            # Assign the volume to the product under the current cycle and line.
            cycle_data[cycle][line][product] = volume

        return cycle_data  


    def load_volume_tank(self, filepath):

        df_read = pd.read_csv(filepath, dtype={'Cycle': str, 'Tank': int, 'Product': str, 'Volume': float})

        # Create an empty dictionary to store the cycle data.
        cycle_data = {}

        # Iterate over the DataFrame rows.
        for index, row in df_read.iterrows():
            cycle, tank, product, volume = row['Cycle'], row['Tank'], row['Product'], row['Volume']
            
            # Ensure the keys are strings to match the target dictionary structure.
            cycle, tank = str(cycle), int(tank)

            # Initialize nested dictionaries if not already present.
            if cycle not in cycle_data:
                cycle_data[cycle] = {}
            if tank not in cycle_data[cycle]:
                cycle_data[cycle][tank] = {}

            # Assign the volume to the product under the current cycle and line.
            cycle_data[cycle][tank][product] = volume

        return cycle_data      
                            


class DataCycle(DataLocation):
    def __init__(self, data_cycle_loader, name, cycle):
        """
        Initializes the DataCycle class by setting cycle-related volumes and validating them.
        
        :data_cycle_loader: An object responsible for loading cycle data.
        :name: The name of the data location.
        :cycle: The specific cycle to manage and validate.
        """
        super().__init__(name)  # Call the parent class constructor
        
        # Ensure all necessary keys and structures are present in VolExist
        self._populate_vol_exist(data_cycle_loader)
        
        # Set cycle-related attributes from data_cycle_loader
        self.Cycle = cycle
        self.CycleVolIn2 = data_cycle_loader.VolIn[cycle]
        self.CycleVolOut2 = data_cycle_loader.VolOut[cycle]
        self.CycleVolExist = data_cycle_loader.VolExist[cycle]
        self.CycleVolLineFlows = data_cycle_loader.VolLineFlows[
            data_cycle_loader.VolLineFlows['Cycle'] == int(cycle)]
        self.CycleStart = self.CycleVolLineFlows['Datetime'].min()

        # Perform data validations
        self.validation_volume()
        self.validation_tanks()
        self.validation_lineFlows()

    def _populate_vol_exist(self, data_cycle_loader):
        """
        Ensures all necessary keys and structures are present in the VolExist dictionary.
        """
        for cycle_key in data_cycle_loader.VolExist:
            for topology_id in self.topo_i:
                for level in self.topo_i[topology_id]:
                    for product in self.topo_i[topology_id][level]:
                        if topology_id not in data_cycle_loader.VolExist[cycle_key]:
                            data_cycle_loader.VolExist[cycle_key][topology_id] = {product: 0}

    def validation_volume(self):
        """
        Validates the volumes for the cycle, checking for inconsistencies or negative values.
        """
        self._validate_volume_data(self.CycleVolExist, 'Exist') # Creates an object volums_exist
        self._validate_volume_data(self.CycleVolOut2, 'Out') # Creates an object volums_out
        self._validate_volume_data(self.CycleVolIn2, 'In') # Creates an object volums_in

        self._calculate_net_volume()

    def _validate_volume_data(self, data, label):
        """
        Helper method to validate volume data and accumulate volumes by product.
        This function dynamically creates an attribute to the DataCycle object. The name of the attribute is determined by the label parameter, 
        ensuring that different aspects of the cycle's volume (exist, in, out) data can be stored and accessed separately. 
        This flexibility allows the class to handle multiple types of volume data in a structured and consistent manner.
        Example: data = {'13': {'54': 6.0, '62': 58.0}, '14': {'62': 24.0, 'A': 144.0}, '15': {'A': 196.0, 'D': 79.0}, 
                         '16': {'54': 127.0}, '17': {'A': 156.0}, '18': {'54': 12.0, '62': 60.0, 'A': 381.0, 'D': 43.0}, 
                         '19': {'A': 389.0, 'D': 56.0}, '1A': {'A': 52.0, 'D': 9.0}, '20': {'54': 35.0, '62': 30.0}, 
                         '2A': {'62': 13.0}
                        }
                 
                 product_to_sum  = {'D', '62', 'A', '54'}
                 product_volumes = {'A': 1318.0, '54': 180.0, 'D': 187.0, '62': 185.0}
                 Creates an object self.volumes_out = {'A': 1318.0, '54': 180.0, 'D': 187.0, '62': 185.0}
        """
        product_to_sum = {prod for page in data.values() for prod in page}
        product_volumes = {key: sum(page.get(key, 0) for page in data.values()) for key in product_to_sum}
        setattr(self, f'volumes_{label.lower()}', product_volumes)

    def _calculate_net_volume(self):
        """
        Calculates and validates the net volume from existing, incoming, and outgoing volumes.
        Example:
        volumes_exist = {'54': 36.0, '62': 105.0, 'D': 90.0, 'A': 159.0}
        volumes_in    = {'54': 324.0, '62': 365.0, 'D': 213.0, 'A': 1418.0}
        volumes_out   = {'54': 180.0, '62': 185.0, 'D': 187.0, 'A': 1318.0}

        validation_volume:
            Exist      In     Out    Net
        54   36.0   324.0   180.0  180.0
        62  105.0   365.0   185.0  285.0
        D    90.0   213.0   187.0  116.0
        A   159.0  1418.0  1318.0  259.0
        """
        df = pd.DataFrame([self.volumes_exist, self.volumes_in, self.volumes_out]).T
        df.columns = ['Exist', 'In', 'Out']
        df['Net'] = df['Exist'] + df['In'] - df['Out']

        setattr(self, f'volumes_net', df)
        
        if (df['Net'] < 0).any():
            raise ValueError("Net volume contains negative values!")

    def validation_tanks(self):
        """
        Validates tank volume data. Currently, this method prepares data for validation.
        Example:
        original_dict = 
        {310: {'A': 86.0}, 311: {'A': 5.0}, 312: {'D': 40.0}, 316: {'D': 4.0}, 317: {'A': 31.0}, 330: {'A': 5.0}, 331: {'A': 4.0}, 
         332: {'A': 5.0}, 333: {'A': 6.0}, 334: {'A': 7.0}, 336: {'A': 4.0}, 337: {'D': 46.0}, 338: {'A': 6.0}, 350: {'54': 4.0}, 
         351: {'54': 5.0}, 352: {'54': 6.0}, 353: {'54': 7.0}, 354: {'54': 14.0}, 360: {'62': 7.0}, 361: {'62': 10.0}, 
         371: {'62': 10.0}, 373: {'62': 10.0}, 374: {'62': 68.0}, 313: {'A': 0.0}, 314: {'A': 0.0}, 339: {'A': 0.0}, 363: {'62': 0.0}, 
         370: {'54': 0.0}, 375: {'62': 0.0}, 376: {'62': 0.0}
         }

        transformed_dict = 
        {310: ['A'], 311: ['A'], 312: ['D'], 313: ['A'], 314: ['A'], 316: ['D'], 317: ['A'], 330: ['A'], 331: ['A'], 332: ['A'], 
        333: ['A'], 334: ['A'], 336: ['A'], 337: ['D'], 338: ['A'], 339: ['A'], 350: ['54'], 351: ['54'], 352: ['54'], 353: ['54'], 
        354: ['54'], 360: ['62'], 361: ['62'], 363: ['62'], 370: ['54'], 371: ['62'], 373: ['62'], 374: ['62'], 375: ['62'], 376: ['62']
        }
        """
        original_dict = self.CycleVolExist
        self.transformed_dict = {key: list(value.keys()) for key, value in sorted(original_dict.items())}

    def validation_lineFlows(self):
        """
        Validates line flows by summarizing hourly volumes per line and product, then calculates net volumes
        by comparing these sums against existing volumes. Raises an error if net volumes are negative.

        df:
        Product        In       Out   Exist      Net
        54        324.007   179.020   36.0   180.987
        62        364.983   181.969   105.0  288.014
        A         1417.993  1316.337  159.0  260.656
        D         212.153   186.499   90.0   115.654
        """
        
        '''
        Group by Line and Product, then sum hourly volumes
        Example: grouped_flows
        Line Product  Hourly_Vol
        0    01       A    1417.993
        1    01       D     212.153
        2    02      54     324.007
        3    02      62     364.983
        4    13      54       6.000
        5    13      62      57.630
        6    14      62      23.540
        7    14       A     143.975
        8    15       A     195.507
        9    15       D      79.002
        10   16      54     126.528
        11   17       A     155.323
        12   18      54      11.500
        13   18      62      59.026
        14   18       A     380.992
        15   18       D      42.997
        16   19       A     388.496
        17   19       D      55.500
        18   1A       A      52.044
        19   1A       D       9.000
        20   20      54      34.992
        21   20      62      29.274
        22   2A      62      12.499
        '''
        grouped_flows = self.CycleVolLineFlows.groupby(['Line', 'Product'])['Hourly_Vol'].sum().reset_index()

        # Determine flow direction (In/Out) based on Line value and add as a new column
        grouped_flows['Type'] = num.where(grouped_flows['Line'].isin(['01', '02']), 'In', 'Out')

        '''
         Pivot the DataFrame to have Products as rows and the sum of In/Out volumes as columns
         pivot_flows:
         Product        In        Out
         54             324.007   179.020
         62             364.983   181.969
          A             1417.993  1316.337
          D             212.153   186.499
        '''
        pivot_flows = grouped_flows.pivot_table(index='Product', columns='Type', values='Hourly_Vol', aggfunc='sum', fill_value=0).reset_index()

        # Merge existing volume data into the pivoted DataFrame
        df_existing_volumes = pd.DataFrame.from_dict(self.volumes_exist, orient='index', columns=['Exist']).reset_index().rename(columns={'index': 'Product'})
        final_df = pivot_flows.merge(df_existing_volumes, on='Product', how='left').fillna(0)

        ''' 
        # Calculate net volume and add as a new column
        Example: final_df
        Product        In       Out        Exist      Net
        54             324.007   179.020   36.0       180.987
        62             364.983   181.969   105.0      288.014
        A              1417.993  1316.337  159.0      260.656
        D              212.153   186.499   90.0       115.654
        '''
        final_df['Net'] = final_df['Exist'] + final_df.get('In', 0) - final_df.get('Out', 0)

        # Optionally, set the result as an attribute for further use or inspection
        setattr(self, f'volumes_net_flows', final_df)

        # Check for negative net volumes and raise an error if found
        if (final_df['Net'] < 0).any():
            raise ValueError("Net volume contains negative values!")
        
    def print_details(self):
            print("\n------- volumes_exist -------")
            print(self.volumes_exist)
            print("\n------- volumes_in -------")
            print(self.volumes_in)
            print("\n------- volumes_out -------")
            print(self.volumes_out)
            print("\n------- volumes_net -------")
            print(self.volumes_net)
            print("\n------- volumes_net_flows -------")
            print(self.volumes_net_flows)
 

class DataOptimization(DataCycle):
    def __init__(self, data_cycle_loader, name, cycle):
        
        # Call the constructor of the parent class
        super().__init__(data_cycle_loader, name, cycle)
        
        self.T     = max(self.CycleVolLineFlows['Time']) - min(self.CycleVolLineFlows['Time']) + 1
        self.Time  = list(range(self.T))
        self.index = self.function_index(self.Time, self.topo_i, self.topo_o)
                            
        inputs_ = {}
        inputs_['index']             = self.index
        inputs_["Cycle"]             = self.Cycle
        inputs_['CycleVolIn2']       = self.CycleVolIn2
        inputs_['CycleVolOut2']      = self.CycleVolOut2
        inputs_['CycleVolExist']     = self.CycleVolExist
        inputs_['CycleVolLineFlows'] = self.CycleVolLineFlows
        inputs_['CycleStart']        = self.CycleStart
        inputs_['Bounds']            = self.Bounds
        inputs_['Capacity']          = self.Capacity
        inputs_['Tanks']             = self.Tanks
        inputs_['Tanks_to_use']      = self.Tanks
        inputs_['T']                 = self.T
        inputs_['Time']              = self.Time
        
        self.inputs = inputs_
        
    def function_index(self, Time, topo_i, topo_o):

        ret = {}
        #-------------------------------------------------------------------------------------------------------------              
        #
        # 
        topo_x = {}
        Tanks_ = list(set(list(topo_i.keys()) + list(topo_o.keys())))
        for tank in Tanks_:
            prods = []
            if tank in topo_i:
                for line in topo_i[tank]:
                    for prod in topo_i[tank][line]:
                        prods.append(prod)
            if tank in topo_o:
                for line in topo_o[tank]:
                    for prod in topo_o[tank][line]:
                        prods.append(prod)
            prods = set(prods)        
            for prod in prods:
                topo_x[tank] = {prod:0}

        #-------------------------------------------------------------------------------------------------------------              
        #
        #   
        i_index = []
        for tank in topo_i:
            for line in topo_i[tank]:
                for product in topo_i[tank][line]:
                    for time in Time:
                        i_index.append((tank, line, product, time))

        ret['i'] = i_index
        #-------------------------------------------------------------------------------------------------------------              
        #
        #    
        o_index = []
        for tank in topo_o:
            for line in topo_o[tank]:
                for product in topo_o[tank][line]:
                    for time in Time:
                        o_index.append((tank, line, product, time))  

        ret['o'] = o_index
        #-------------------------------------------------------------------------------------------------------------              
        #
        #            
        x_index = []
        for tank in topo_x:
            for prod in topo_x[tank]:
                for time in Time:
                    x_index.append((tank, prod, time))            


        ret['x'] = x_index
        #-------------------------------------------------------------------------------------------------------------              
        #
        #            
        mi_index = [] 
        tups = set([tup[1:3] for tup in i_index if tup[3] == 0])
        for tup in tups:
            for time in Time:
                mi_index.append(tup + (time,))

        mo_index = [] 
        tups = set([tup[1:3] for tup in o_index if tup[3] == 0])
        for tup in tups:
            for time in Time:
                mo_index.append(tup + (time,))    


        ret['mi'] = mi_index
        ret['mo'] = mo_index
        #-------------------------------------------------------------------------------------------------------------              
        #
        #              
        li_index = []
        lst = set([tup[1:4] for tup in i_index])
        for tup in lst:
            li_index.append(tup)   

        lo_index = []
        lst = set([tup[1:4] for tup in o_index])
        for tup in lst:
            lo_index.append(tup) 

        ret['li'] = li_index
        ret['lo'] = lo_index
        #-------------------------------------------------------------------------------------------------------------              
        #
        #              
        ti_index = []
        lst = set([tup[0:4] for tup in i_index])
        for tup in lst:
            ti_index.append(tup) 

        to_index = []
        lst = set([tup[0:4] for tup in o_index])
        for tup in lst:
            to_index.append(tup) 

        ret['ti'] = ti_index
        ret['to'] = to_index
        #-------------------------------------------------------------------------------------------------------------              
        #
        #              
        tlpo_index = []
        lst = set([tup[0:3] for tup in o_index])
        for tup in lst:
            tlpo_index.append(tup)

        ret['tlpo'] = tlpo_index

        #-------------------------------------------------------------------------------------------------------------              
        #
        #              
        tank_index = []
        Tanks_ = list(set(list(topo_i.keys()) + list(topo_o.keys())))
        for tank in Tanks_:
            tank_index.append(tank,)

        ret['tank'] = tank_index

        #-------------------------------------------------------------------------------------------------------------              
        #
        #    
        return (ret)    
        
class DataOptimizations():
    
    def __init__(self, data_cycle_loader):
        
        optimizations_ = {}
        optimizations_['031'] = DataOptimization(data_cycle_loader, "ATJ", "031")
        optimizations_['041'] = DataOptimization(data_cycle_loader, "ATJ", "041")
        optimizations_['051'] = DataOptimization(data_cycle_loader, "ATJ", "051")
        optimizations_['061'] = DataOptimization(data_cycle_loader, "ATJ", "061")
        optimizations_['071'] = DataOptimization(data_cycle_loader, "ATJ", "071") 
        optimizations_['081'] = DataOptimization(data_cycle_loader, "ATJ", "081")
        
        self.optimizations = {'ATJ': optimizations_}
        
    def getOptimization(self, name, cycle):
        return self.optimizations[name][cycle]