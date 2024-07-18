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
from pathlib import Path
from ast import literal_eval
import ast  # Import abstract syntax trees to safely evaluate strings as Python expressions


class DataInputProcessing():

    def __init__(self):
        
        self.dir_raw               = Path('input_raw')
        self.dir_stage             = Path('input_staging')
        self.dir_location          = Path('input_location')
        self.dir_cycle             = Path('input_cycle')  
        self.filename_lineSchedule  = 'TRM SxcheduleData.csv'
        self.filename_tankInventory = 'TFInvSample.csv'
        
        self.staging()
        
        self.create_dimension_tanks()
        self.create_dimension_products()
        self.create_dimension_lines()
        
        self.create_fact_lineSchedule()
        self.create_fact_tank()
 
    def staging(self):
        
        """
        This function does some basic file staging before the main processing.
        Two main files are handled: 'lineSchedule' and 'tankInventory'. These files may come with different names and they are placed in the input_raw directory. The files are altered as follows:


        - 'lineSchedule':  (i) Adjusts batch codes by mapping the product to a simplified version, 
                           and (ii) keep only one cycle.
        - 'tankInventory': (i) Maps the product codes (similar to above), and (ii) corrects negative volumes.

        Note:
            Files paths are derived from instance attributes and depend on the correct setting
            of `filename_lineSchedule` and `filename_tankInventory`.
            The resulting files are placed into input_staging directory.
        """
             
        self.stage_lineSchedule()
        self.stage_tankInventory()
        
    
    def stage_lineSchedule(self):
        
        # Setup the file paths
        input_path  = self.dir_raw / self.filename_lineSchedule
        output_path = self.dir_stage / self.filename_lineSchedule
        
        # Read the CSV file
        df = pd.read_csv(input_path)
            
        # Change 1: Replace the product in the Batch column using the mapping
        def replace_component(value): # Function to replace middle component
            parts = value.split('-')
            middle_component = parts[1]
            parts[1] = self.mapping(middle_component)
            return '-'.join(parts)
        # Apply the function to the DataFrame
        df['Batch'] = df['Batch'].apply(replace_component)
        
        # Change 2: Keep only one cycle (this may change)
        extracted_data = df['Batch'].str.extract(r'(\w+)-(\w+)-(\w+)') # Extract and append client, product, and cycle information from the 'Batch' column
        extracted_data.columns = ['Client', 'Product', 'Cycle_']
        df = pd.concat([df, extracted_data], axis=1)
        # Truncate the 'Cycle_' column to get the first two characters and filter rows
        df['Cycle'] = df['Cycle_'].str[:2]
        df = df[df['Cycle'] == '52']  # Filter for rows where 'Cycle' is '52'
        # Drop unnecessary columns from the DataFrame
        df.drop(['Client', 'Product', 'Cycle_', 'Cycle'], axis=1, inplace=True)
        
        # Save the updated DataFrame
        df.to_csv(output_path, index=False)
        
    def stage_tankInventory(self):
        
        # Setup the file paths
        input_path  = self.dir_raw / self.filename_tankInventory
        output_path = self.dir_stage / self.filename_tankInventory
        
        # Read the CSV file
        df = pd.read_csv(input_path)
        
        # Change 1: Apply the mapping function to the Product column
        df['Product'] = df['Product'].apply(self.mapping)
   
        # Change 2: If Volume is negative replace it with zero
        df = df.dropna(subset=['Product'])
        df['Volume'] = df['Volume'].where(df['Volume'] >= 0, 0)
        
        # Save the output
        df.to_csv(output_path, index=False)
    
    def mapping(self, code):
        if code in ['A2', 'A3', 'A4', 'A5', 'M3', 'M4', 'V4']:
            return 'A'
        elif code in ['D2', 'D3', 'D4']:
            return 'D'
        elif code in ['51', '56']:
            return '54'
        elif code in ['96']:
            return '62'
        else:
            return code    

    def create_dimension_tanks(self):
        
        """
        In this function we build dim_tanks, i.e., a dimension table that will be the reference point for Tanks. 
        The operation is primarily based on the tankInventory file. We perform the following operations:
        
        - Rename some of the existing columns
        - Divide the volumes by 1000 and round to zero
        - Add a 'LineIn' and 'LineOut' component describing the lines that go in and out of every Tank
        
        """

        # Setup file paths
        input_file   = self.dir_stage / self.filename_tankInventory
        output_file  = self.dir_location / 'dim_tanks.csv'

        # Ensure the output directory exists
        #base_output_path.mkdir(parents=True, exist_ok=True)

        # Read only the necessary columns from file
        df = pd.read_csv(input_file, usecols=['Tank', 'Type', 'Product', 
                                              'Type', 'Out of Service',
                                              'Low Level', 'Normal Fill', 'Max Capacity'])
        
        # Rename columns
        df.rename(columns={'Low Level': 'Bottom',
                          'Normal Fill': 'Working',
                          'Max Capacity': 'Maximum'}, inplace=True)
        
        # Columns to modify
        columns_to_modify = ['Bottom', 'Working', 'Maximum']

        # Divide by 100 and round to three decimal places
        df[columns_to_modify] = df[columns_to_modify].div(1000).round(0)
        df[columns_to_modify] = df[columns_to_modify] + 80 # 30 Here you can change the capacity
        
        # Function to read and group lines by tank
        def group_lines(file_path, line_column_name):
            df_lines = pd.read_csv(file_path, usecols=['Tank', 'Line'])
            grouped = df_lines.groupby('Tank')['Line'].agg(list).reset_index()
            return grouped.rename(columns={'Line': line_column_name})
        
        # Process and merge line out data
        line_out_path = self.dir_location / 'topoO_ATJ.csv'
        line_out_data = group_lines(line_out_path, 'LineOut')
        df = df.merge(line_out_data, on='Tank', how='left')

        # Process and merge line in data
        line_in_path = self.dir_location / 'topoI_ATJ.csv'
        line_in_data = group_lines(line_in_path, 'LineIn')
        df = df.merge(line_in_data, on='Tank', how='left')

        # Save output
        df.to_csv(output_file, index=False)
        print("File saved to:", output_file)
            
    
    def create_dimension_products(self):

        '''
         In this function we build dim_products, i.e., a dimension table that will be the reference point for Products. 
         The operation is based on both the tankInventory (file1) and the lineSchedule (file2) files. 
         We take the unique Products from file1 (df1), the unique products from the inbound file2 (df2) and the unique products from
         outbount file2 (df3). Then we aggregate everything together to get a complete picture of all the products that show up in the 
         input files
        '''
        
        # Setup file paths
        input_file1 = self.dir_stage / self.filename_tankInventory
        input_file2 = self.dir_stage / self.filename_lineSchedule
        output_file = self.dir_location / 'dim_products.csv'
        
        # Read file
        tank_df = pd.read_csv(input_file1)
        
        # Get the unique products from file1 and put them into a dataframe
        df1 = pd.DataFrame(tank_df['Product'].unique(), columns=['Product'])
        df1['TankFile'] = True

        # Read file2 and extract product details
        line_df = pd.read_csv(input_file2)
        line_df[['Client', 'Product', 'Cycle_']] = line_df['Batch'].str.extract(r'(\w+)-(\w+)-(\w+)')

        # Break line_df into two pieces to produce df2 & df3. First get df2
        line_in_df  = line_df[line_df['Line'].isin(['01', '02'])]
        df2 = pd.DataFrame(line_in_df['Product'].unique(), columns=['Product'])
        df2['LineIn'] = True

        # And then df3
        line_out_df = line_df[~line_df['Line'].isin(['01', '02'])]
        df3 = pd.DataFrame(line_out_df['Product'].unique(), columns=['Product'])
        df3['LineOut'] = True

        # Concatenate the three dataframes and group by 'Product' to aggregate boolean columns
        df = pd.concat([df1, df2, df3])
        df = df.groupby('Product', as_index=False).max()

        # Replace NaN with False in boolean columns
        df.fillna(False, inplace=True)

        # save output
        df.to_csv(output_file, index=False)
        
        print(f"File saved to: {output_file}")


    def create_dimension_lines(self):
        
        '''
         In this function we build dim_lines, i.e., a dimension table that will be the reference point for Lines. 
         The operation is primarily based on the lineSchedule file. We perform the following operations:
        
        - Add a 'Tank' component describing the tanks associated with each Line
        '''
        
        # Set up file paths
        input_path = Path('input_staging') / self.filename_lineSchedule
        output_path = Path('input_location') / 'dim_lines.csv'
        
        # Ensure the output directory exists
        #output_path.parent.mkdir(parents=True, exist_ok=True)

        # Read file
        df = pd.read_csv(input_path)

        # Extract and rename columns directly in the DataFrame
        df[['Client', 'Product', 'Cycle_']] = df['Batch'].str.extract(r'(\w+)-(\w+)-(\w+)')

        # Group by 'Line' and aggregate unique 'Products'
        result_df = df.groupby('Line')['Product'].agg(lambda x: sorted(set(x))).reset_index()

        # Attach a column indicating the Tanks associated with the Line
        df_tank = pd.read_csv('input_location/topoO_ATJ.csv', usecols=['Tank', 'Line'])
        grouped_tanks = df_tank.groupby('Line')['Tank'].agg(list).reset_index()
        
        # Merge back to the original tanks DataFrame
        result_df = result_df.merge(grouped_tanks, on='Line', how='left')
        
        # Create a "Type" column
        def getType(line):
            if line in ['01', '02']:
                return 'In'
            else:
                return 'Out'
        result_df['Type'] = result_df['Line'].apply(getType)
        
        # Save to CSV, excluding the index
        result_df.to_csv(output_path, index=False)

        # Print the result or confirmation message
        print(f"File saved to: {output_path}")
        
    def create_fact_lineSchedule(self):
        
        # Set up file paths
        input_path = Path('input_staging') / self.filename_lineSchedule
        output_path = Path('input_cycle') / 'fact_lineSchedule.csv'
        
        # Read the files
        df = pd.read_csv(input_path)

        #--------------------------------------------------------------------------------
        # Step 1: Create the 'Superbatch' column
        #
        # Initialize the SuperBatch column with NaN
        df['SuperBatch'] = num.nan

        # Placeholder for the current SuperBatch value
        current_super_batch = None

        # Iterate over the DataFrame rows
        for index, row in df.iterrows():
            if row['Event Type'] == 'SUPERSTART':
                # Update the current SuperBatch value
                current_super_batch = row['Batch']
            elif row['Event Type'] == 'SUPERSTOP':
                # Clear the current SuperBatch value
                current_super_batch = None

            # Assign the current SuperBatch value to the row
            #df.at[index, 'SuperBatch'] = current_super_batch
            # Convert the 'SuperBatch' column to string type
            df['SuperBatch'] = df['SuperBatch'].astype(str)
            # Now, assigning the string should not raise an error
            df.at[index, 'SuperBatch'] = current_super_batch

        df.to_csv('input_cycle/processing/df_step1.csv')

        #--------------------------------------------------------------------------------
        # Step 2: Replace the RATECHANGE
        #
        # Initialize a list to hold the new rows
        new_rows = []

        # Iterate through RATECHANGE rows and create new BATCHSTART and BATCHEND rows
        for index, row in df.iterrows():

            if row['Event Type'] == 'RATECHANGE':
                # Update the current SuperBatch value
                new_start_row = row.copy()
                new_end_row = row.copy()
                # Set the Event Type for the new rows
                new_start_row['Event Type'] = 'BATCHSTART'
                #new_start_row['Batch'] = new_start_row['Batch'] + '-1'
                new_end_row['Event Type'] = 'BATCHSTOP'
                #new_end_row['Batch'] = new_end_row['Batch'] + '-1'
                # Add the new rows to the list
                new_rows.append(new_end_row)
                new_rows.append(new_start_row)
            elif (row['Event Type'] == 'BATCHSTART') | (row['Event Type'] == 'BATCHSTOP'):
                # Clear the current SuperBatch value
                new_rows.append(row)

        # Create a DataFrame from the new rows
        new_rows_df = pd.DataFrame(new_rows)    
        new_rows_df.rename(columns={'Event Date Time': 'Start Date Time'}, inplace=True)
        # Shift the 'Event Date Time' column upwards by one
        new_rows_df['Stop Date Time'] = new_rows_df['Start Date Time'].shift(-1)

        df_start = new_rows_df[new_rows_df['Event Type'] == 'BATCHSTART'].copy()

        #df_start['Start'] = pd.to_datetime(df_start['Start Date Time'])
        #df_start['Stop']   = pd.to_datetime(df_start['Stop Date Time'])

        # Convert 'Stop Date Time' column to datetime using a specified format
        df_start['Start'] = pd.to_datetime(df_start['Start Date Time'], format="%m/%d/%Y %I:%M:%S %p")
        df_start['Stop'] = pd.to_datetime(df_start['Stop Date Time'], format="%m/%d/%Y %I:%M:%S %p")



        df_start.to_csv('input_cycle/processing/df_step2.csv')
        
        #--------------------------------------------------------------------------------
        # Step 3: Expand
        #
        # Define the original data with 'Start' and 'End' columns, possibly with additional rows
        original_df = df_start

        # Function to expand time range for a given start and end time
        def expand_time_range(start_time, end_time):
            start = pd.to_datetime(start_time)
            end = pd.to_datetime(end_time)

            # Ensure that 'end' is always after 'start'
            if end < start:
                return pd.DataFrame({'Timestamp': [], 'Hours': []})

            # Initialize variables
            timestamps = []
            hours = []

            # Truncate the start time to the beginning of its hour
            start_truncated = start.replace(minute=0, second=0, microsecond=0)

            # Handle same-hour case directly
            if start.hour == end.hour and start.date() == end.date():
                duration_minutes = (end - start).total_seconds() / 60
                fraction_of_hour = round(duration_minutes / 60, 2)
                timestamps.append(start_truncated.strftime('%m/%d/%Y %H:%M'))
                hours.append(fraction_of_hour)
            else:
                # Calculate the fraction for the first partial hour if start minute is not zero
                if start.minute != 0:
                    first_hour_fraction = round((60 - start.minute) / 60, 2)
                    timestamps.append(start_truncated.strftime('%m/%d/%Y %H:%M'))
                    hours.append(first_hour_fraction)
                    start_truncated += pd.Timedelta(hours=1)  # Move to the next hour

                # Fill full hours between the adjusted start and the hour before the end
                end_truncated = end.replace(minute=0, second=0, microsecond=0)
                while start_truncated < end_truncated:
                    timestamps.append(start_truncated.strftime('%m/%d/%Y %H:%M'))
                    hours.append(1.0)
                    start_truncated += pd.Timedelta(hours=1)

                # Handle last partial hour if there's any minute in end time
                if end.minute != 0:
                    last_hour_fraction = round(end.minute / 60, 2)
                    timestamps.append(end_truncated.strftime('%m/%d/%Y %H:%M'))
                    hours.append(last_hour_fraction)

            return pd.DataFrame({'Timestamp': timestamps, 'Hours': hours})

        # Initialize an empty DataFrame to store the expanded data
        expanded_data = pd.DataFrame()

        # Iterate over each row in the original DataFrame to generate the expanded rows
        for _, row in original_df.iterrows():
            expanded_rows = expand_time_range(row['Start'], row['Stop'])

            # Copy the additional columns' data for each expanded row
            for col in original_df.columns.difference(['Start', 'Stop']):
                expanded_rows[col] = row[col]

            # Append the expanded rows to the expanded_data DataFrame
            expanded_data = pd.concat([expanded_data, expanded_rows], ignore_index=True)

        # Add Client, Product, Cycle
        expanded_data['Volume'] = (expanded_data['Hours'] * expanded_data['BPH']).round(0)
        temp = expanded_data['SuperBatch'].str.extract(r'(\w+)-(\w+)-(\w+)')
        temp.columns = ['Client', 'Product', 'Cycle_']
        expanded_data = pd.concat([expanded_data, temp], axis=1)
        expanded_data['Cycle'] = expanded_data['Cycle_'].str[:2]
        expanded_data.to_csv('input_cycle/processing/df_step3.csv')

        #--------------------------------------------------------------------------------
        # Step 4: Group data
        #
        grouped_data = expanded_data.groupby(['Cycle', 'Timestamp', 'Line', 'Product']).agg({
            'Volume': 'sum',
            'Start Date Time': 'min',
            'Stop Date Time': 'min'
        }).reset_index()

        # Sorting the resulting DataFrame by 'Volume' (descending) and 'Timestamp' (ascending)
        grouped_data = grouped_data.sort_values(by=['Cycle', 'Timestamp', 'Line', 'Start Date Time'], ascending=[True, True, True, True])

        grouped_data.to_csv('input_cycle/processing/df_step4.csv')
        
        #--------------------------------------------------------------------------------
        # Step 5: Fold data
        #
        # Function to apply to each group
        def tag_similar_next_row(group):
            group['Fold'] = (group['Timestamp'] == group['Timestamp'].shift(-1)).astype(int)
            return group

        # Apply the function to each group defined by 'Line' and 'Cycle'
        grouped_data = grouped_data.groupby(['Line', 'Cycle']).apply(tag_similar_next_row).reset_index(drop=True)
        grouped_data = grouped_data[grouped_data['Fold'] == 0]

        grouped_data.to_csv('input_cycle/processing/df_step5.csv')
        
        #--------------------------------------------------------------------------------
        # 
        #
        # Renaming the columns
        grouped_data.rename(columns={'Timestamp': 'Datetime'}, inplace=True)
        grouped_data['Datetime'] = pd.to_datetime(grouped_data['Datetime'])
        grouped_data['Volume'] = grouped_data['Volume'] / 1000
        grouped_data['Diff'] = 1
        grouped_data['Hourly_Vol'] = grouped_data['Volume']
        grouped_data['Datetime_min'] = 1
        grouped_data['Time'] = 1
        # Define the new order of columns
        new_order = ['Datetime', 'Volume', 'Diff', 'Cycle', 'Hourly_Vol', 'Product', 'Line', 'Datetime_min', 'Time']

        grouped_data['Time'] = grouped_data.groupby('Cycle')['Datetime'].rank(method = 'dense', ascending=True) 

        # Find the minimum datetime value
        grouped_data['Datetime_min'] = grouped_data.groupby('Cycle')['Datetime'].transform('min')

        grouped_data['Time'] = (grouped_data['Datetime'] - grouped_data['Datetime_min']).dt.total_seconds() / 3600 + 1

        # Reorder the columns by indexing with the new order
        grouped_data = grouped_data[new_order]


        # Ensure all 'Cycle' column values are strings with leading zeros if necessary
        grouped_data['Cycle'] = grouped_data['Cycle'].astype(str).str.zfill(3)

        grouped_data.to_csv('input_cycle/processing/df_step6.csv')
        
        # Save to file
        grouped_data.to_csv(output_path, index=False)

        print(f"File saved to: {output_path}")
            
    def create_fact_tank(self):
        
        '''
         In this function we build fact_tanks, i.e., a fact table for Lines. 
         The facts will be the 'Volume' that the Tank holds at the beginning of the Cycle.
         The table is built directly from the tankInventory file similarly to the dim_tanks table.
        '''
        
        # Set up file paths
        input_path = Path('input_staging') / self.filename_tankInventory
        output_path = Path('input_cycle') / 'fact_tanks.csv'
        
        # Read input file
        df = pd.read_csv(input_path)
        
        # Get the 'Volume' and divide it by 1000
        df['Cycle'] = '052'
        df = df[['Cycle', 'Tank', 'Product', 'Volume']]
        df['Volume'] = df['Volume'].where(df['Volume'] >= 0, 0)
        df['Volume'] = (df['Volume'] / 1000).round(0) + 50 # 20

        # Output file
        df.to_csv(output_path, index=False)

        print(f"File saved to: {output_path}")
        
class DataLocation:
    
    def __init__(self):
        
        self.dim_tanks    = self.load_csv('input_location/dim_tanks.csv')
        self.dim_lines    = self.load_csv('input_location/dim_lines.csv')
        self.dim_products = self.load_csv('input_location/dim_products.csv')
        
        # Directly assign the dictionaries returned from the function
        self.topo_i, self.topo_o = self.load_topology()
    
    def load_csv(self, filepath):
        try:
            # Read the CSV file
            df_read = pd.read_csv(filepath, encoding="ISO-8859-1")
            return df_read
        
        except Exception as e:
            print(f"An error occurred in load_csv: {e}")
            return {}
    
    def load_topology(self):
        '''
        Example: 
         topo_i:
        {310: {'01': {'A': 0}}, 311: {'01': {'A': 0}}, 312: {'01': {'D': 0}}, 313: {'01': {'A': 0}}, 314: {'01': {'A': 0}}, 
         316: {'01': {'D': 0}}, 317: {'01': {'A': 0}}, 330: {'01': {'A': 0}}, 331: {'01': {'A': 0}}, 332: {'01': {'A': 0}}, 
         333: {'01': {'A': 0}}, 334: {'01': {'A': 0}}, 336: {'01': {'A': 0}}, 337: {'01': {'D': 0}}, 338: {'01': {'A': 0}}, 339: {'01': {'A': 0}}, 
         350: {'02': {'54': 0}}, 351: {'02': {'54': 0}}, 352: {'02': {'54': 0}}, 353: {'02': {'54': 0}}, 354: {'02': {'54': 0}}, 
         360: {'02': {'62': 0}}, 361: {'02': {'62': 0}}, 363: {'02': {'62': 0}}, 370: {'02': {'54': 0}}, 371: {'02': {'62': 0}}, 
         373: {'02': {'62': 0}}, 374: {'02': {'62': 0}}, 375: {'02': {'62': 0}}, 376: {'02': {'62': 0}}}

         topo_o:
         {310: {'13': {'A': 0}, '14': {'A': 0}, '15': {'A': 0}, '17': {'A': 0}, '18': {'A': 0}, '19': {'A': 0}, '1A': {'A': 0}}, 
          311: {'13': {'A': 0}, '14': {'A': 0}, '15': {'A': 0}, '17': {'A': 0}, '18': {'A': 0}, '19': {'A': 0}, '1A': {'A': 0}}, 
          312: {'13': {'D': 0}, '14': {'D': 0}, '15': {'D': 0}}
        
        '''
        try:
            # Read the CSV file
            df = self.dim_tanks

            # Convert 'Lineout' from string to list
            df['Lineout'] = df['LineOut'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])
            df['Linein'] = df['LineIn'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])

            # Initialize an empty dictionary
            tank_dict = {}

            # Iterate through each row to populate the dictionary
            b = {}
            for index, row in df.iterrows():
                tank = row['Tank']
                product = row['Product']
                lines = row['Lineout']  # Assuming this is already a list, convert if it's a string

                a = {}
                if pd.notna(product):  # Check if product is not NaN before entering the loop
                    for line in lines:
                        line = str(line).zfill(2)
                        a[line] = {product: 0}
                b[tank] = a  
                
            c = {}
            for index, row in df.iterrows():
                tank = row['Tank']
                product = row['Product']
                lines = row['Linein']  # Assuming this is already a list, convert if it's a string

                a = {}
                if pd.notna(product):  # Check if product is not NaN before entering the loop
                    for line in lines:
                        line = str(line).zfill(2)
                        a[line] = {product: 0}
                c[tank] = a 
                
            return c, b
        
        except Exception as e:
            print(f"An error occurred in load_topology: {e}")
            return {}        
            
    def validation_master(self):
        
        print("\n***** Start of Validation Procedures *****")
        self.validation_level0_dim()
        self.validation_level1_dim_tanks()
        self.validation_level1_dim_lines()
        self.validation_level1_dim_products()
        self.validation_level2_dim_1()
        self.validation_level2_dim_2()
        
    def validation_level0_dim(self):
        """
        Validates whether key tables contain the expected columns and raises a value error otherwise.
        """
        required_columns = ['Tank', 'Type', 'Product', 'Working', 'LineOut', 'LineIn']
        df = self.dim_tanks
        missing_columns = [column for column in required_columns if column not in df.columns]
        if missing_columns:
            raise ValueError("Error (validation_level0): Columns are missing from dim_tanks.")
            
        required_columns = ['Product']
        df = self.dim_products
        missing_columns = [column for column in required_columns if column not in df.columns]
        if missing_columns:
            raise ValueError("Error (validation_level0): Columns are missing from dim_products.")
            
        required_columns = ['Line', 'Product']
        df = self.dim_lines
        missing_columns = [column for column in required_columns if column not in df.columns]
        if missing_columns:
            raise ValueError("Error (validation_level0): Columns are missing from dim_lines.")

        print("validation_level0_dim was executed successfully")    
    
    def validation_level1_dim_tanks(self):
        """
        Validates that the 'Tank' column in the 'dim_tanks' DataFrame is suitable for use as a primary key.
        The column must contain unique non-null values only.
        Raises:
            ValueError: If non-null values in the 'Tank' column are not unique.
                        If certain columns have null values
        """
        col = 'Tank'
        if not self.dim_tanks[col].is_unique or self.dim_tanks[col].isnull().any():
            raise ValueError("Error (validation_level1_dim_tanks): The primary key is not unique or contains NULL values.")
            
        columns_to_check = ['Type', 'Product', 'Working', 'LineOut', 'LineIn']
        nan_counts = self.dim_tanks[columns_to_check].isna().sum()
        if nan_counts.any():
            raise ValueError("Error (validation_level1_dim_tanks): NULL values detected")

        print("validation_level1_dim_tanks was executed successfully")          
      
    def validation_level1_dim_products(self):
        """
        Validates that the 'Product' column in the 'dim_products' DataFrame is suitable for use as a primary key.
        The column must contain unique non-null values only.
        Raises:
            ValueError: If non-null values in the 'Product' column are not unique.
        """
        col = 'Product'
        is_ok = self.dim_products[col].is_unique or self.dim_products[col].isnull().any()
        if not is_ok:
            raise ValueError("Error (validation_level1_dim_products): The primary key is not unique or contains NULL values.")
        
        print("validation_level1_dim_products was executed successfully") 
          
    def validation_level1_dim_lines(self):
        """
        Validates that the 'Line' column in the 'dim_lines' DataFrame is suitable for use as a primary key.
        The column must contain unique non-null values only.
        Raises:
            ValueError: If non-null values in the 'Lines' column are not unique.
                        If certain columns have null values
        """
  
        col = 'Line'
        is_ok = self.dim_lines[col].is_unique or self.dim_lines[col].isnull().any()
        if not is_ok:
            raise ValueError("Error (validation_level1_dim_lines): The primary key is not unique or contains NULL values.")
            
        columns_to_check = ['Product']
        nan_counts = self.dim_lines[columns_to_check].isna().sum()
        if nan_counts.any():
            raise ValueError("Error (validation_level1_dim_lines): NULL values detected")
        
        print("validation_level1_dim_lines was executed successfully") 
            
    def validation_level2_dim_1(self):
        
        if not list(self.topo_i.keys()) == list(self.topo_o.keys()):
            raise ValueError("Error: Input and Output topology keys are not the same.")
            
        tanks = self.dim_tanks['Tank'].tolist()
        if not list(self.topo_i.keys()) == sorted(tanks):
            raise ValueError("Error: Topology and Tank keys do not match.")  

        print("validation_level2_dim_1 was executed successully") 
        
    def validation_level2_dim_2(self):

        # Read the CSV
        df_lines = self.dim_lines

        # Clean the 'Tank' data by removing brackets and splitting correctly
        #df_lines['Tank'] = df_lines['Tank'].str.replace('[\[\]]', '', regex=True)  # Remove brackets
        df_lines['Tank'] = df_lines['Tank'].str.replace('\\[\\]', '', regex=True)  # Remove brackets
        df_lines['Tank'] = df_lines['Tank'].apply(lambda x: x.split(', ') if pd.notna(x) else [])

        # Explode the 'Tank' column
        df_lines_v2 = df_lines.explode('Tank')

        # Read the CSV
        df_tanks = self.dim_tanks

        # Convert join columns to integers
        df_lines_v2['Tank'] = pd.to_numeric(df_lines_v2['Tank'], errors='coerce').fillna(-1).astype(int)
        df_tanks['Tank'] = pd.to_numeric(df_tanks['Tank'], errors='coerce').fillna(-1).astype(int)
        left_merged_df = pd.merge(df_lines_v2, df_tanks, on='Tank', how='left')

        # Group by 'Line' and aggregate 'Product_y' into a list, excluding NaNs
        result_df = left_merged_df.groupby('Line').agg({
            'Product_x': 'first',  # Assuming Product_x should just be taken from the first occurrence
            'Product_y': lambda x: list(x.dropna().unique())  # Remove NaNs and get unique Product_y values
        }).reset_index()
        
        def convert_to_list(s):
            try:
                return ast.literal_eval(s)
            except ValueError:
                return []

        # Apply the function to convert columns
        result_df['Product_x'] = result_df['Product_x'].apply(convert_to_list)
        result_df['Product_y'] = result_df['Product_y'].apply(lambda x: x if isinstance(x, list) else convert_to_list(x))
        
        def is_subset(row):
            set_x = set(row['Product_x'])
            set_y = set(row['Product_y'])
            return set_x.issubset(set_y)

        # Apply this function to each row
        result_df['is_subset'] = result_df.apply(is_subset, axis=1)        
        #print("result_df:\n", result_df)

        print("validation_level2_dim_2 was executed successfully")

    def print(self):
        print("\n------- Topology I -------")
        print(self.topo_i)
        print("\n------- dim Tanks -------")
        print(self.dim_tanks)
        print("\n------- dim Lines -------")
        print(self.dim_lines)
        print("\n------- dim Products -------")
        print(self.dim_products)
        

class DataCycle(DataLocation):
    
    def __init__(self):
        """
        Initializes the DataCycle class by setting cycle-related volumes and validating them.
        
        :data_cycle_loader: An object responsible for loading cycle data.
        :name: The name of the data location.
        :cycle: The specific cycle to manage and validate.
        """
        super().__init__()  # Call the parent class constructor
        
        self.fact_tanks        = self.load_csv('input_cycle/fact_tanks.csv')
        self.fact_LineSchedule = self.load_csv('input_cycle/fact_LineSchedule.csv')
        
        self.CycleStart = self.fact_LineSchedule['Datetime'].min()
        self.T          = int(max(self.fact_LineSchedule['Time']) - min(self.fact_LineSchedule['Time']) + 1)
        self.Time       = list(range(self.T))
        
    def load_csv(self, filepath):
        # Read the CSV file
        df_read = pd.read_csv(filepath, encoding="ISO-8859-1")
        return df_read    

    def validation_master(self):
        
        # Call the parent class's show method first
        super().validation_master()
        # Now add additional functionality
        self.validation_level1_fact_tanks()
        self.validation_level2_fact_1()
        self.validation_level2_fact_2()
        self.validation_level2_fact_3()
        self.validation_level2_fact_4()
        print("***** Validation Procedures executed successfully ***** \n")
    
    def validation_level1_fact_tanks(self):
        """
        Validates that the 'Tank' column in the 'fact_tanks' DataFrame is suitable for use as a primary key.
        The column must contain unique non-null values only.
        Raises:
            ValueError: If non-null values in the 'Tank' column are not unique.
                        If certain columns have null values
        """
        col = 'Tank'
        if not self.fact_tanks[col].is_unique or self.fact_tanks[col].isnull().any():
            raise ValueError("Error (validation_level1_fact_tanks): The primary key is not unique or contains NULL values.")
            
        columns_to_check = ['Tank', 'Volume']
        nan_counts = self.fact_tanks[columns_to_check].isna().sum()
        if nan_counts.any():
            raise ValueError("Error (validation_level1_fact_tanks): NULL values detected")
        
        print("validation_level1_fact_tanks was executed successfully")

    # def _validate_volume_data(self, data, label):
    #     """
    #     Helper method to validate volume data and accumulate volumes by product.
    #     This function dynamically creates an attribute to the DataCycle object. The name of the attribute is determined by the label parameter, 
    #     ensuring that different aspects of the cycle's volume (exist, in, out) data can be stored and accessed separately. 
    #     This flexibility allows the class to handle multiple types of volume data in a structured and consistent manner.
    #     Example: data = {'13': {'54': 6.0, '62': 58.0}, '14': {'62': 24.0, 'A': 144.0}, '15': {'A': 196.0, 'D': 79.0}, 
    #                      '16': {'54': 127.0}, '17': {'A': 156.0}, '18': {'54': 12.0, '62': 60.0, 'A': 381.0, 'D': 43.0}, 
    #                      '19': {'A': 389.0, 'D': 56.0}, '1A': {'A': 52.0, 'D': 9.0}, '20': {'54': 35.0, '62': 30.0}, 
    #                      '2A': {'62': 13.0}
    #                     }
                 
    #              product_to_sum  = {'D', '62', 'A', '54'}
    #              product_volumes = {'A': 1318.0, '54': 180.0, 'D': 187.0, '62': 185.0}
    #              Creates an object self.volumes_out = {'A': 1318.0, '54': 180.0, 'D': 187.0, '62': 185.0}
    #     """
    #     product_to_sum = {prod for page in data.values() for prod in page}
    #     product_volumes = {key: sum(page.get(key, 0) for page in data.values()) for key in product_to_sum}
    #     setattr(self, f'volumes_{label.lower()}', product_volumes)

    def validation_level2_fact_1(self):
        """
        Validates line flows by summarizing hourly volumes per line and product, then calculates net volumes
        by comparing these sums against existing volumes. Raises an error if net volumes are negative.

        The resulting dataframe looks like that:
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

        grouped_flows = self.fact_LineSchedule.groupby(['Line', 'Product'])['Hourly_Vol'].sum().reset_index()
        # Determine flow direction (In/Out) based on Line value and add as a new column
        grouped_flows['Type'] = num.where(grouped_flows['Line'].isin(['01', '02']), 'In', 'Out')
        pivot_flows = grouped_flows.pivot_table(index='Product', columns='Type', values='Hourly_Vol', aggfunc='sum', fill_value=0).reset_index()

        '''
         Pivot the DataFrame to have Products as rows and the sum of In/Out volumes as columns
         pivot_flows:
         Product        In        Out
         54             324.007   179.020
         62             364.983   181.969
          A             1417.993  1316.337
          D             212.153   186.499
        '''
        
        exist = self.fact_tanks.groupby(['Product'])['Volume'].sum().reset_index().rename(columns={'Volume': 'Exist'})
        final_df = exist.merge(pivot_flows, on='Product', how='left').fillna(0)

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
        setattr(self, f'fact_summary_net', final_df)

        # Check for negative net volumes and raise an error if found
        if (final_df['Net'] < 0).any():
            print(final_df)
            raise ValueError("Net volume contains negative values!")
        
        print("validation_level2_fact_1 was completed successfully. The flow summary is: \n", final_df)
            
    def validation_level2_fact_2(self):

        # Aggregate 'Hourly_Vol' by 'Time', 'Line', 'Product' and compute in/out sign.
        # 'num' should be 'np' for NumPy which is typically used for such operations.
        a = self.fact_LineSchedule
        a = (a.groupby(['Time', 'Line', 'Product'])
            .agg(Hourly_Vol=('Hourly_Vol', 'sum'))
            .reset_index())

        # Determine 'Type' and 'Sign', then adjust 'Hourly_Vol'.
        a['Sign'] = num.where(a['Line'].isin(['01', '02']), 1, -1)
        a['Hourly_Vol'] *= a['Sign']  # Apply 'Sign' directly in the computation.

        # Sum 'Hourly_Vol' by 'Time' and 'Product', compute cumulative volume.
        a = (a.groupby(['Time', 'Product'])
            .agg(Hourly_Vol=('Hourly_Vol', 'sum'))
            .reset_index())
        a['Cumulative_Vol'] = a.groupby('Product')['Hourly_Vol'].cumsum()

        # Merge with 'fact_summary_net', compute 'Net', and handle missing values.
        a = a.merge(self.fact_summary_net, on='Product', how='left').fillna(0)
        a['Net'] = a['Exist'] + a['Cumulative_Vol']

        if (a['Net'] < 0).any():
            #print(a[a['Net'] < 0])
            raise ValueError("Error (class_cycle.validation_level2_fact_2)")
        
        print("validation_level2_fact_2 was executed successfully")


    def validation_level2_fact_3(self):
        """
        """
        unique_col1 = set(self.dim_tanks['Tank'].unique())
        unique_col2 = set(self.fact_tanks['Tank'].unique())
        if not unique_col1 == unique_col2:
            raise ValueError("Error (class_cycle.validation_level2_fact_3): The primary key are not the same between dim and fact tank.")
        
        print("validation_level2_fact_3 was executed successfully")  


    def validation_level2_fact_4(self):
        """
        """
        a = self.dim_tanks.merge(self.fact_tanks, on="Tank", how="left")
        filtered_df = a[a['Volume'] > a['Working']]
        if not filtered_df.empty:
            raise ValueError("Error (class_cycle.validation_level2_fact_4): ")
        
        #print(a)
        print("validation_level2_fact_4 was executed successfully")       


class DataOptimization(DataCycle):
    def __init__(self):
        
        # Call the constructor of the parent class
        super().__init__()
        super().validation_master()
        
        self.index = self.function_index(self.Time, self.topo_i, self.topo_o)
                            
        inputs_ = {}
        inputs_['index']              = self.index
        inputs_['dim_tanks']          = self.dim_tanks
        inputs_['dim_products']       = self.dim_products
        inputs_['dim_lines']          = self.dim_lines
        inputs_['fact_LineSchedule']  = self.fact_LineSchedule
        inputs_['fact_tanks']         = self.fact_tanks
        inputs_['CycleStart']         = self.CycleStart
        inputs_['T']                  = self.T
        inputs_['Time']               = self.Time
        
        self.inputs = inputs_
        
    def function_index(self, Time, topo_i, topo_o):

        ret = {}

        '''
        This is a list of tuples that will be used later as a blueprint for defining the state space of the model variables.
        For example, a sample of i_index can be:
        [(310, '01', 'A', 0), (310, '01', 'A', 1), (310, '01', 'A', 2), (310, '01', 'A', 3)]
        ''' 
        ret['i'] = [(tank, line, product, time) for tank in topo_i for line in topo_i[tank] for product in topo_i[tank][line] for time in Time]

        '''
        For example, a sample of o_index can be:
        [(310, '13', 'A', 0), (310, '13', 'A', 1), (310, '13', 'A', 2), (310, '13', 'A', 3), (310, '13', 'A', 4), 
         (310, '13', 'A', 5), (310, '13', 'A', 6), (310, '13', 'A', 7), (310, '13', 'A', 8), (310, '13', 'A', 9)]
        '''    
        ret['o'] = [(tank, line, product, time) for tank in topo_o for line in topo_o[tank] for product in topo_o[tank][line] for time in Time]

        '''
        For example, a sample of x_index can be:
        [(310, 'A', 0), (310, 'A', 1), (310, 'A', 2), (310, 'A', 3), (310, 'A', 4), (310, 'A', 5), 
         (310, 'A', 6), (310, 'A', 7), (310, 'A', 8), (310, 'A', 9)]
         
        topo_x:
        {310: 'A', 311: 'A', 312: 'D', 313: 'A', 314: 'A', 316: 'D', 317: 'A', 330: 'A', 331: 'A', 332: 'A', 
         333: 'A', 334: 'A', 336: 'A', 337: 'D', 338: 'A', 339: 'A', 350: '54', 351: '54', 352: '54', 353: '54', 
         354: '54', 360: '62', 361: '62', 363: '62', 370: '54', 371: '62', 373: '62', 374: '62', 375: '62', 376: '62'}
        '''
        combined_dict = {**topo_i, **topo_o}
        topo_x = {}
        for outer_key, middle_dict in combined_dict.items():
            for middle_key, inner_dict in middle_dict.items():
                for inner_key in inner_dict.keys():
                    topo_x[outer_key] = inner_key
                    break  # Since we only need the first key, we can break after finding it

        ret['x'] = [(tank, topo_x[tank], time) for tank in topo_x for time in Time]

        '''
        For example, a sample of mi_index can be:
        [('02', '62', 0), ('02', '62', 1), ('02', '62', 2), ('02', '62', 3), ('02', '62', 4), ('02', '62', 5), 
         ('02', '62', 6), ('02', '62', 7), ('02', '62', 8), ('02', '62', 9)]
        '''
        tups = {(tup[1], tup[2]) for tup in ret['i'] if tup[3] == 0}
        ret['mi'] = [(line, product, time) for line, product in tups for time in Time]

        '''
        For example, a sample of mo_index can be:
        [('14', 'A', 0), ('14', 'A', 1), ('14', 'A', 2), ('14', 'A', 3), ('14', 'A', 4), ('14', 'A', 5), 
         ('14', 'A', 6), ('14', 'A', 7), ('14', 'A', 8), ('14', 'A', 9)]
        '''
        tups = {(tup[1], tup[2]) for tup in ret['o'] if tup[3] == 0}
        ret['mo'] = [(line, product, time) for line, product in tups for time in Time]

        '''
        For example, a sample of li_index can be:
        [('01', 'D', 136), ('01', 'A', 27), ('02', '62', 185), ('01', 'D', 145), ('01', 'A', 36), ('02', '54', 63), 
         ('02', '54', 72), ('01', 'A', 45), ('02', '54', 81), ('01', 'A', 54)]
        '''            
        ret['li'] = list({tup[1:4] for tup in ret['i']})

        '''
        For example, a sample of lo_index can be:
        [('13', '54', 141), ('19', '62', 126), ('14', 'A', 111), ('15', 'A', 0), ('13', 'A', 39), ('17', 'A', 138), 
         ('14', '54', 68), ('19', '54', 31), ('17', '54', 95), ('18', 'D', 86)]
        ''' 
        ret['lo'] = list({tup[1:4] for tup in ret['o']})

        '''
        For example, a sample of ti_index can be:
        [(353, '02', '54', 121), (360, '02', '62', 71), (374, '02', '62', 157), (354, '02', '54', 71), (317, '01', 'A', 54), 
         (313, '01', 'A', 85), (331, '01', 'A', 78), (312, '01', 'D', 68), (314, '01', 'A', 87), (330, '01', 'A', 43)]
        '''               
        ret['ti'] = list(set(ret['i']))
        
        '''
        For example, a sample of to_index can be:
        [(310, '19', 'A', 9), (361, '2A', '62', 117), (363, '2A', '62', 168), (371, '17', '62', 183), (333, '17', 'A', 139), 
         (373, '2A', '62', 38), (338, '14', 'A', 93), (350, '19', '54', 95), (351, '17', '54', 16), (337, '13', 'D', 94)]
        ''' 
        ret['to'] = list(set(ret['o']))

        '''
        For example, a sample of tlpo_index can be:
        [(339, '17', 'A'), (310, '13', 'A'), (337, '18', 'D'), (363, '14', '62'), (352, '20', '54'), (311, '17', 'A'), 
         (334, '17', 'A'), (363, '17', '62'), (360, '20', '62'), (376, '17', '62')]
        '''              
        ret['tlpo'] = list({tup[:3] for tup in ret['o']})

        '''
        For example, a sample of tank_index can be:
        [310, 311, 312, 313, 314, 316, 317, 330, 331, 332]
        '''              
        tank_index = []
        Tanks_ = list(set(list(topo_i.keys()) + list(topo_o.keys())))
        for tank in Tanks_:
            tank_index.append(tank,)

        ret['tank'] = tank_index

        return (ret)    