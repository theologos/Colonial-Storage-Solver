from datetime import datetime
import pandas as pd
import time
import csv
from datetime import datetime, timedelta

class DataAnalysis:
        
    def process_tank_volumes():
        """
        Static method to process tank volume data from a schedule stored as a CSV file.

        This function reads a CSV file containing a schedule. 
        It then determines whether each line is an 'In' or 'Out' type based on specific line codes. 
        The volume data is scaled by a factor of 1000. 
        It groups and aggregates the data by tank and type to calculate the total 'In' and 'Out' volumes for each tank. 
        The resulting dataframe is written to a new CSV file and returned.

        Test:
            The function will be tested if a file "results/opt_schedule_1.csv" is available. Then use:
            import pandas as pd
            df = DataAnalysis.process_tank_volumes()

        Returns:
            A pandas DataFrame with three columns: 'Tank', 'Volume_In', and 'Volume_Out'.
        """

        # Read the csv file
        schedule = pd.read_csv("results/opt_schedule_1.csv")

        # Define the 'Type' based on 'Line' values
        schedule['Type'] = schedule['Line'].map({'01': 'In', '02': 'In'}).fillna('Out')

        # Scale the 'Volume' column
        schedule['Volume'] *= 1000

        # Group and aggregate data
        df = (schedule.groupby(['Tank', 'Type'])
                      .agg({'Volume': 'sum'})
                      .unstack(fill_value=0)
                      .reset_index())

        # Flatten the MultiIndex in columns
        df.columns = ['Tank', 'Volume In', 'Volume Out']
        
        # Add Tank numbering
        df.insert(0, '#', range(1, 1 + len(df)))

        # Write the results
        df.to_csv("results/opt_analysis_tanks.csv", index=False)

        # Return the dataframe
        return df


    @staticmethod
    def process_line_volumes():
        """
        Static method to process line data from a schedule stored as a CSV file.

        This function reads a CSV file containing a schedule. 
        It groups the data by 'Line', 'Product', and 'Tank', and then aggregates the volumes.
        The volume data is then scaled by a factor of 1000 and rounded to the nearest whole number.
        The resulting DataFrame is saved to a new CSV file in the 'results' directory.

        Test:
            The function will be tested if a file "results/opt_schedule_1.csv" is available. Then use:
            import pandas as pd
            df = DataAnalysis.process_line_volumes()

        Returns:
            pandas.DataFrame: A DataFrame with columns 'Line', 'Product', 'Tank', and 'Volume',
            where 'Volume' represents the aggregated and scaled volume data for each group.
        """
        # Read the csv file and perform operations
        df = (pd.read_csv("results/opt_schedule_1.csv")
             .groupby(['Line', 'Product', 'Tank'])
             .agg({'Volume':'sum'})
             .reset_index()
             .assign(Volume=lambda x: (x['Volume'] * 1000).round(0))
             )

        # Write the results
        df.to_csv("results/opt_analysis_lines.csv", index=False)

        # Return
        return df

    
    @staticmethod
    def inputFlow(cycle):

        s = pd.read_csv('results/VolLineFlows_3.csv')
        s = s[s['Cycle'] == int(cycle)]

        s = (s.groupby(['Line', 'Product'])
         .agg({'Hourly_Vol':'sum'})
         .reset_index()
         .assign(Volume=lambda x: x['Hourly_Vol'] * 1000)
         )
        # Round 'column_name' to 2 decimal places
        s['Hourly_Vol'] = s['Hourly_Vol'].round(0)
        s.rename(columns={'Hourly_Vol': 'Volume'}, inplace=False)
        s = s.drop(columns=['Hourly_Vol'])

        # Write the results
        s.to_csv("results/opt_analysis_inputFlow.csv", index=False)

        # Return
        return s
    
    @staticmethod
    def store_variables(ID, inputs):
   
        x_index, i_index, o_index, x, i, o, p, q, CycleVolIn2, CycleVolOut2, CycleVolExist, CycleStart, T = (
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
                inputs['CycleStart'],
                inputs['T']
        )

        #-------------------------------------------------------------------------------------------------------------                    
        # Define the dataframe called "data"
        #    
        data = []
        tups = [tup for tup in x_index]
        for tup in tups:
            tank = tup[0]; prod = tup[1]; j = tup[2];
            if x[tank, prod, j].X > 0:
                row = [tank, prod, j, x[tank, prod, j].X]
                data.append(row)
        df = pd.DataFrame(data, columns=['Tank','Product', 'Time', 'Volume'])
        df.to_csv("results/X_" + str(ID) + ".csv") 

        data = []
        tups = [tup for tup in i_index]
        for tup in tups:
            tank = tup[0]; line = tup[1]; prod = tup[2]; j = tup[3];
            if p[tank, line, prod, j].X > 0 and i[tank, line, prod, j].X > 0:
                row = [tank, line, prod, j, i[tank, line, prod, j].X]
                data.append(row)
        df = pd.DataFrame(data, columns=['Tank', 'Line', 'Product', 'Time', 'Volume'])
        df.to_csv("results/i_" + str(ID) + ".csv") 

        tups = [tup for tup in o_index]
        for tup in tups:
            tank = tup[0]; line = tup[1]; prod = tup[2]; j = tup[3];
            if q[tank, line, prod, j].X > 0 and o[tank, line, prod, j].X > 0:
                row = [tank, line, prod, j, o[tank, line, prod, j].X]
                data.append(row)
        df = pd.DataFrame(data, columns=['Tank', 'Line', 'Product', 'Time', 'Volume'])
        df.to_csv("results/o_" + str(ID) + ".csv") 

        #-------------------------------------------------------------------------------------------------------------                    
        # Write CycleVolExist to a file
        #
        data3 = []
        tups = [tup for tup in x_index]
        for tup in tups:
            tank = tup[0]; prod = tup[1]; j = tup[2];
            row = [j, tank, prod, x[tank, prod, j].X]
            data3.append(row)  

        CycleVolExist2 = {}
        tups = [tup[1:4] for tup in data3 if tup[0] == T-1]
        for tup in tups:
            CycleVolExist2[tup[0]] = {tup[1]: round(tup[2])}

        # Sample dictionary
        my_dict = CycleVolExist2

        #-------------------------------------------------------------------------------------------------------------                    
        # Write CycleVolExist to a file
        #
        # Write dictionary to CSV
        with open("results/opt_CycleVolExist_" + str(ID) + ".csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(['Key', 'Value'])
            for key, value in my_dict.items():
                writer.writerow([key, value])
        

    @staticmethod
    def schedule(ID, inputs):
   
        x_index, i_index, o_index, x, i, o, p, q, CycleVolIn2, CycleVolOut2, CycleVolExist, CycleStart, T = (
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
                inputs['CycleStart'],
                inputs['T']
        )

        #-------------------------------------------------------------------------------------------------------------                    
        # Define the dataframe called "data"
        #    
        data = []
        tups = [tup for tup in i_index]
        for tup in tups:
            tank = tup[0]; line = tup[1]; prod = tup[2]; j = tup[3];
            if p[tank, line, prod, j].X > 0 and i[tank, line, prod, j].X > 0:
                row = [tank, line, prod, j, i[tank, line, prod, j].X]
                data.append(row)

        tups = [tup for tup in o_index]
        for tup in tups:
            tank = tup[0]; line = tup[1]; prod = tup[2]; j = tup[3];
            if q[tank, line, prod, j].X > 0 and o[tank, line, prod, j].X > 0:
                row = [tank, line, prod, j, o[tank, line, prod, j].X]
                data.append(row)

        #-------------------------------------------------------------------------------------------------------------                    
        # df:    A clean dataframe based on data
        # Flag:  Defines the first element of a batch
        # Flag2: Defines the batch number
        #
        df = pd.DataFrame(data, columns=['Tank', 'Line', 'Product', 'Time', 'Volume'])
        df.to_csv("results/df_" + str(ID) + ".csv") 
        df['Time1']=df.groupby(['Tank', 'Line', 'Product'])['Time'].shift(-1) - df['Time']
        df['Time2']=df['Time'] - df.groupby(['Tank', 'Line', 'Product'])['Time'].shift(1)  

        def flag_df(df):
            if (pd.isnull(df['Time2']) or df['Time2'] > 1):
                return 'First'
        df['Flag'] = df.apply(flag_df, axis = 1)

        s = 0;
        a = df["Flag"].tolist()
        b = []
        for k in a:
            if k == 'First':
                s = s + 1
            b.append(s)
        df['Flag2'] = b

        #-------------------------------------------------------------------------------------------------------------                    
        # Data 1: The solution Line volumes
        #
        data1 = df.groupby(['Line', 'Product']).agg({'Volume': 'sum'})
        data1 = data1.reset_index()
        data1 = data1.rename(columns = {'Volume':'Volume_solution'})

        #-------------------------------------------------------------------------------------------------------------                    
        # Data 2: The required Line volumes
        #
        data2 = []
        for line in CycleVolIn2:
            for prod in CycleVolIn2[line]:
                row = [line, prod, CycleVolIn2[line][prod]]
                data2.append(row)

        for line in CycleVolOut2:
            for prod in CycleVolOut2[line]:
                row = [line, prod, CycleVolOut2[line][prod]]
                data2.append(row)

        data2 = pd.DataFrame(data2, columns = ['Line', 'Product', 'Volume_actual'])

        #-------------------------------------------------------------------------------------------------------------                    
        # Data 2: Join with Data 1 to get the difference
        #
        data2 = data2.join(data1.set_index(['Line', 'Product']), on=['Line', 'Product'])
        data2['Volume_diff'] = data2['Volume_solution'] - data2['Volume_actual']

        #-------------------------------------------------------------------------------------------------------------                    
        # Schedule: Joing df with data 2 
        #
        schedule = df.merge(data2, on=['Line', 'Product'], how='left')

        #-------------------------------------------------------------------------------------------------------------                    
        # Schedule: Add "Mode" to schedule
        #
        df = schedule
        def flag_df(df):
            if (df['Line'] in ['01', '02']):
                return 'In'
            else:
                return 'Out'
        df['Mode'] = df.apply(flag_df, axis = 1)
        schedule = df

        df3      = schedule
        schedule = df3.groupby(['Tank', 'Line', 'Product', 'Flag2']).agg({'Time': ['min', 'max'], 'Volume': 'sum'})
        schedule.columns = ['_'.join(col) for col in schedule.columns.values]
        schedule = schedule.reset_index()

        #-------------------------------------------------------------------------------------------------------------                    
        # Schedule: Add Start_Date_Time + Finish_Date_Time
        #
        #'2023-01-22 02:41:00'
        #date = datetime.datetime(2023, 1, 22, 2, 0, 0)
        date = datetime.strptime(CycleStart, '%Y-%m-%d %H:%M:%S')
    
        s = 0;
        a = schedule["Time_min"].tolist()
        b = []
        for k in a:
            b.append(date + timedelta(hours=k))
        schedule['Start_Date_Time'] = b

        s = 0;
        a = schedule["Time_max"].tolist()
        b = []
        for k in a:
            b.append(date + timedelta(hours=k+1))
        schedule['Finish_Date_Time'] = b

        #-------------------------------------------------------------------------------------------------------------                    
        # 
        #
        b = pd.DataFrame(CycleVolExist.items(), columns=['Tank', 'Product'])
        def flag_df(d):
            return list(d.values())[0]
        b['Volume_Exists'] = b['Product'].apply(flag_df)
        schedule = schedule.merge(b[['Tank', 'Volume_Exists']], on=['Tank'], how='left')
        schedule = schedule.rename(columns={'Volume_sum': 'Volume'})

        schedule['Rank'] = range(1, len(schedule) + 1)
        schedule = schedule[['Rank'] + [col for col in schedule if col != 'Rank']]
        schedule['Volume'] = schedule['Volume'].round(3)
        #-------------------------------------------------------------------------------------------------------------                    
        # 
        #
        schedule.to_csv("results/opt_schedule_" + str(ID) + ".csv", index=False)

        return(schedule) 