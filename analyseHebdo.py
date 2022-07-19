# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 06:51:22 2022

@author: patal
@organization: BeamcutSystems
@project: weeklyAnalysis

"""
# Modules and Libraries *****************************************

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sb
#import plotly.graph_objects as go
import plotly.express as px
from plotly.offline import plot
import datetime
import time


# to copy in the streamlit/config.toml file in the [theme] section:
# [theme]
# base="dark"
# primaryColor="#f9a800"
# font="monospace"

# Custom Functions **********************************************
def convert_df(df):
   return df.to_csv() #.encode('utf-8')

# Get time in seconds to readable time string
def s_to_hms (sec):
    neg = False
    
    if sec < 0:
        neg = True
        sec = -1*sec
        
    mm, ss = divmod(sec,60)
    hh, mm = divmod(mm,60)
    
    if round(hh)<10:
        hh='0'+str(round(hh))
    else:
        hh= str(round(hh))
        
    if round(mm)<10:
        mm='0'+str(round(mm))
    else:
        mm= str(round(mm))
        
    if round(ss)<10:
        ss='0'+str(round(ss))
    else:
        ss= str(round(ss))
    
    if neg == True:
        time_str = '-'+ hh + ':' + mm + ':' + ss
    else:
        time_str = hh + ':' + mm + ':' + ss
        
    return time_str

# Get time in ms to readable time string
def ms_to_hms(ms):
    seconds = ms / 1000
    time_str = s_to_hms(seconds)
    return time_str
    
    
# Min-Max Normalization over a numeric column:
def minmaxNorm(processData, colName):  
    # copy the data
    processData_min_max_scaled = processData.copy() 
    # apply normalization techniques by Column
    processData_min_max_scaled[colName] = ((processData_min_max_scaled[colName] - processData_min_max_scaled[colName].min()) / (processData_min_max_scaled[colName].max() - processData_min_max_scaled[colName].min()))*100    
    # return normalized column
    return processData_min_max_scaled[colName]


def writeCycleId(date_string):
     # this function generates a cycle string ID based on CYCLE=STARTED timestamp for traceability:
    cycleID = date_string.replace('-', '').replace('/', '').replace(' ','').replace(':', '')
    return cycleID

def generateCycleDetails(processData):
    
    # Find STARTS and ENDS and store their indexes:
    starts = processData.index[processData['cycle'] == 'STARTED'].tolist()
    # ends = processData.index[processData['cycle'] == 'END'].tolist()
    # STARTED - RSR0003 is generally caught but not END - RSR0003
    # Assuming a raw bar is from one STARTED RSR0003 to the next one (excluded):
    ends = starts[1:] # remove first start
    ends.append(len(processData)) # add end of data
    ends = [index-1 for index in ends] # end of cycle considered row before START

    # slice for cycle - between start and end - identify cycle by started timestamp
    for i in range(0,len(starts)):
        lower_limit = starts[i]
        upper_limit = ends[i]
        # cycle ID
        processData.loc[lower_limit : upper_limit, 'cycleID'] = writeCycleId(processData.date[lower_limit])   
        # Cycle Purpose
        cycle_programs = processData.loc[lower_limit : upper_limit, 'program'].values
        # Check if JB is called in range
        if ('JB_PROG' or 'RSR0003') in cycle_programs:
            # Set cycle purpose:
            processData.loc[lower_limit : upper_limit, 'cyclePurpose'] = 'rawProcessing'
        elif 'RSR0001' in cycle_programs:
            # Set cycle id to 
            processData.loc[lower_limit : upper_limit, 'cyclePurpose'] = 'SendRobotHome'
        elif 'RSR0002' in cycle_programs:
            processData.loc[lower_limit : upper_limit, 'cyclePurpose'] = 'CalibratingPlasma'
        elif ('RSR0005' or 'RSR006') in cycle_programs:
            processData.loc[lower_limit : upper_limit, 'cyclePurpose'] = 'RSR0005'
        elif 'RSR0007' in cycle_programs:
            processData.loc[lower_limit : upper_limit, 'cyclePurpose'] = 'GoMaintenance'
        elif 'RSR0008' in cycle_programs:
            processData.loc[lower_limit : upper_limit, 'cyclePurpose'] = 'CloseForNight'
        else:
            processData.loc[lower_limit : upper_limit, 'cyclePurpose'] = 'UnrecognizedOrAborted'
        
    # define cycleID as category data type:
    processData['cycleID'] = processData['cycleID'].astype('category') 
    # define cycle Purpose as category data type:
    processData['cyclePurpose'] = processData['cyclePurpose'].astype('category')
    # define program names as categories:
    processData['program'] = processData['program'].astype('category')
    
    return processData

def generateProdData(prodData):
    # Data Engineering
    prodData = prodData.where(prodData.Status=='COMPLETED')
    prodData = prodData.dropna() # index original
    prodData['piece_weight'] = prodData.apply(lambda row:(row.spec_length/25.4/12)* row.weight_lbs_per_foot, axis=1 )
    #prodData['bridge_m'] = prodData['bridge_s'].apply(lambda chrono: round(chrono/60))
    
    return prodData

def generateProdData_Daily(prodData):
    #prodDaily = pd.DataFrame()
    prodDaily = prodData.groupby('prod_date').prod_delta_s.sum().reset_index().rename(columns={'prod_delta_s':'activeProd_s'})
    #prodDaily['activeProd_s'] = prodData.groupby('prod_date').prod_delta_s.sum().reset_index()
    
    prodDaily['activeProd_t'] = prodDaily.activeProd_s.apply(lambda chrono: s_to_hms(chrono))
    
    bridgeTotal = prodData.groupby('prod_date').bridge_s.sum().reset_index().rename(columns={'bridge_s':'bridgeTotal_s'})
    prodDaily = pd.merge(prodDaily, bridgeTotal, how='left')

    # Raw bar per day:
    raw_bar_per_day = prodData.groupby('prod_date').actual_length.nunique().reset_index().rename(columns={'actual_length':'rawBars'})
    prodDaily = pd.merge(prodDaily, raw_bar_per_day, how='left')
    
    # Machined Piece Per Day (Output count):
    # production_item_id can repeat itself for multiple pieces in the same raw bar (bar nesting) -> consider duplicates
    piece_per_day = prodData.groupby('prod_date').production_item_id.count().reset_index().rename(columns={'production_item_id':'prodItems'})
    prodDaily = pd.merge(prodDaily, piece_per_day, how='left')
    
    # Linear Footage Daily Stats:
    # production_item_id can repeat itself if multiple pieces in a bar -> raw length must be considered once
    # i.e.  production_item_id must be viewd as a nesting id for a single bar
    singleCompletedId = prodData.drop_duplicates(subset=['production_item_id'], keep='first').reset_index()
    raw_linfootage_per_day = singleCompletedId.groupby('prod_date').actual_length.apply(lambda length: sum(length)/25.4/12).reset_index().rename(columns={'actual_length':'rawFootage'})  
    # Output Linear Footage per day
    out_linfootage_per_day = prodData.groupby('prod_date').spec_length.apply(lambda length: sum(length)/25.4/12).reset_index().rename(columns={'spec_length':'outputFootage'})
    prodDaily = pd.merge(prodDaily, raw_linfootage_per_day, how='left')
    prodDaily = pd.merge(prodDaily, out_linfootage_per_day, how='left')
    
    # Converted footage ratio (handled raw footage vs outputed machined footage):
    prodDaily['ftConversionRatio'] = prodDaily.apply(lambda row: (row.outputFootage / row.rawFootage)*100 , axis=1)
    
    # Average raw length per day
    avg_raw_length_per_day = prodData.groupby('prod_date').actual_length.apply(lambda length: length.mean()/25.4/12).reset_index().rename(columns={'actual_length':'avgRawLength_ft'})
    prodDaily = pd.merge(prodDaily, avg_raw_length_per_day, how='left')    
    
    # Machined weight (lbs) per day
    out_weight_per_day = prodData.groupby('prod_date').piece_weight.sum().reset_index().rename(columns={'piece_weight':'outWeight_lbs'})
    prodDaily = pd.merge(prodDaily, out_weight_per_day, how='left')
    
    return prodDaily

def generateProdData_Hourly(prodDaily):
    #prodHourly = pd.DataFrame()
    
    prodHourly = prodDaily[['prod_date', 'activeProd_s', 'activeProd_t']]
    
    # Handled Raw Feet per Hour (feeded raw feet per hour):
    prodHourly['handleRawFt_h'] = prodDaily.apply(lambda row: round( row.rawFootage / (row.activeProd_s/60/60), 2), axis=1)

    # Raw bars per Hour:
    prodHourly['rawBars_h'] = prodDaily.apply(lambda row: round(row.rawBars / (row.activeProd_s/60/60), 2), axis=1)

    # Machined Piece per Hour:
    prodHourly['pieces_h'] = prodDaily.apply(lambda row: round(row.prodItems / (row.activeProd_s/60/60), 2), axis=1)

    # Output weight per Hour (lbs and tonnage):
    prodHourly['outWeightLbs_h'] = prodDaily.apply(lambda row: row.outWeight_lbs / (row.activeProd_s/60/60), axis=1)
    prodHourly['outWeightTons_h'] = prodHourly['outWeightLbs_h'].apply(lambda weight: round(weight / 2000, 3))
    
    return prodHourly

def generateAgg_Hourly(prodHourly):
    prodHourlyAgg = prodHourly.mean()
    prodHourlyAgg = (prodHourlyAgg.round(decimals = 2)) #.reset_index() #.columns(['Stat', 'weeklyAvg']) #.to_frame()
    numberOfDays = len(prodHourly)
    return prodHourlyAgg, numberOfDays

def generateProcessData(processData):
    # remove pause rows for a realistic timeDelta sum per cycleID
    processData = processData[processData.program != 'PAUSE'] 

    # total process time for each cycleID
    agg_cycle = processData.groupby('cycleID').timeDelta.sum().reset_index().rename(columns={'timeDelta':'cycleTime_ms'})
    agg_cycle = agg_cycle[agg_cycle.cycleTime_ms > 0]
    
    # Process Time Distribution for every cycle by programs
    agg_prog = processData.groupby(['cycleID', 'program']).timeDelta.sum().reset_index().rename(columns={'timeDelta':'progTime_ms'})
    agg_prog = agg_prog[agg_prog.progTime_ms > 0]  
    
    # Cycle time Proportion %
    # total cycle time in agg_cycle
    # total program time for a cycle in agg_prog
    agg_prog = pd.merge(agg_prog, agg_cycle, how='left')
    agg_prog['time%'] = agg_prog.apply(lambda row: round((row.progTime_ms/row.cycleTime_ms)*100, 3), axis=1)

    # Average time per program over all raw bars:
    agg_bars = agg_prog.groupby('program').progTime_ms.mean().reset_index().rename(columns={'progTime_ms':'avgProgTime'})
    agg_bars = agg_bars.dropna()
    # Program Frequency on the call stack amongst all cycles
    overall_prog_freq = processData.program.value_counts(normalize=False).reset_index().rename(columns={'program':'progFreq', 'index':'program'})

    # merge to aggregate
    agg_bars = pd.merge(agg_bars, overall_prog_freq, how='left')

    # Resident Programs Only - Average time per resident program over all raw bars:
    agg_residents = agg_bars[agg_bars.program.str.contains('^E_', regex= True, na=False)]

    return agg_cycle, agg_prog, agg_bars, overall_prog_freq, agg_residents

# def sliceDataTimeframe(data, column, start, end):
#     dataSlice = data.where((data[column] >= start) & (data[column] <= end))
#     dataSlice = dataSlice.dropna() # original indexes
#     return dataSlice

def getDeltas(currentStats, previousStats):
    deltas = currentStats.subtract(previousStats, fill_value = 0)
    return deltas

# def newChangeLog(date, time, descritpion, location, status):
#     return


# Load & Clean Data *****************************************************

# * Load data (and standard Clean) through a function to apply streamlit cache and prevent complete reloading
#@st.cache
def load_data():
    
    # Production Data ***
    # apply Macro_HMIView to csv before reading to processData
    prodData = pd.read_csv('HMIView.csv', sep='\s*;\s*',
                           header=0, encoding='utf-8', engine='python')
    print(prodData.columns.tolist())
    # Datetime object
    #prodData['creation_date'] = pd.to_datetime(prodData['creation_date'])
    #prodData['modification_date'] = pd.to_datetime(prodData['modification_date'])
    prodData['StartDateTime'] = pd.to_datetime(prodData['StartDateTime'])
    prodData['EndDateTime'] = pd.to_datetime(prodData['EndDateTime'])
    #prodData['prod_dateObj'] = pd.to_datetime(prodData['prod_date'])
    # Column names adjustments:
    prodData = prodData.rename(columns={'spceLenght':'spec_length'})
    # Datatypes Adjustments:
    prodData['bridge_s'] = prodData['bridge_s'].astype('float64')
    
    # Process Data ***
    # apply Macro_PLC_log to csv before reading to processData
    processData = pd.read_csv('logcyl.csv')

    # remove all columns labeled 'unused':
    processData = processData.loc[:, (processData.columns !='unused')]
    # Detect Pauses and attribute program name 'pause'
    # To avoid pause time on a task being attributed to that task
    processData.loc[processData.cycle == 'PAUSED', 'program'] = 'PAUSE'
    # Universal program name for all JB_ -> JB_prog (except TRIMCUT)
    processData['program'] = processData['program'].str.replace(r'^JB_(?!TRIMCUT).*', 'JB_PROG', regex=True)
    # datetime object :
    processData['dateObj'] = pd.to_datetime(processData['date'])
    
    return prodData, processData

prodData, processData = load_data()

def loadChangelog():
    changelog = pd.read_csv('Changelog.csv')
    changelog = changelog.dropna(how='all').dropna(axis=1, how='all') # drop rows with all missing values
    return changelog

changelog = loadChangelog()

# Data Engineering **********************************************


# Common ***

#firstDay = str(st.date_input('Premier Jour de la période courante'))
#lastDay = str(st.date_input('Dernier Jour de la période courante'))

# Production week or range: (YYYY-MM-DD)
firstDay = '2022-07-06' #                           select Streamlit
lastDay ='2022-07-17'   #                           select Streamlit

# Production compared - previous week or past range:
offset_firstDay = '2022-06-13' # 12 juin changement temps de mesure
offset_lastDay = '2022-07-04' # 5 jullet modification au plc
    

# Production ***
prodDataClean = generateProdData(prodData)
prodDataSliced = prodDataClean.where((prodDataClean.prod_date >= firstDay) & (prodDataClean.prod_date <= lastDay))
prodDataSliced = prodDataSliced.dropna().reset_index()
prodDataAvailable = False
prev_prodDataAvailable = False

if len(prodDataSliced) > 0:
    prodDataAvailable = True
    
    prodDaily = generateProdData_Daily(prodDataSliced)
    prodHourly = generateProdData_Hourly(prodDaily)
    prodHourlyAgg, numberOfDays = generateAgg_Hourly(prodHourly)
    
    # Production - Previous Period
    prodDataSliced_prev = prodDataClean.where((prodDataClean.prod_date >= offset_firstDay) & (prodDataClean.prod_date <= offset_lastDay))
    prodDataSliced_prev = prodDataSliced_prev.dropna().reset_index()
    
    if len(prodDataSliced_prev) > 0:
        prev_prodDataAvailable = True
        
        prodDaily_prev = generateProdData_Daily(prodDataSliced_prev)
        prodHourly_prev = generateProdData_Hourly(prodDaily_prev)
        prodHourlyAgg_prev, numberOfDays_prev = generateAgg_Hourly(prodHourly_prev)
        
        # Compare Production Averages
        deltas = getDeltas(prodHourlyAgg, prodHourlyAgg_prev)
        
    else:
        missingPreviousProdData_mess = 'Les données de production ne sont pas disponibles pour la prédiode comparative (précédente) sélectionnée.'

else:
     missingProdData_mess = 'Les données de production ne sont pas disponibles pour la prédiode courante sélectionnée.'


# Process ***
processDataClean = generateCycleDetails(processData)
processDataSliced = processDataClean.where((processDataClean.dateObj >= firstDay) & (processDataClean.dateObj <= lastDay))
processDataSliced = processDataSliced.dropna() # original indexes
processDataAvailable = False

if len(processDataSliced) > 0:
    processDataAvailable = True
    
    agg_cycle, agg_prog, agg_bars, overall_prog_freq, agg_residents = generateProcessData(processDataSliced)

else:
    missingProcessData_mess = 'Les données relatives aux programmes sont manquantes pour la période sélectionnée.'

# Visualizations ************************************************

# Building Graphics ***
if processDataAvailable:
    
    # Cycle Time Breakdown by program
    # Interactive figure allowing to unselect Pause and irrelevant program from Cycle Time Breakdown
    timeBreakDownPerCycle = px.bar(agg_prog, 
                                   x='progTime_ms', y='cycleID', 
                                   color='program', orientation='h', barmode='relative',
                                   labels={'progTime_ms':'Temps cumulé (ms)', 'program':'Programme', 'cycleID':'Cycles échantillonnés'}
                                   )
    timeBreakDownPerCycle.update_yaxes(showticklabels=False)
    #plot(timeBreakDownPerCycle)
    
    # Cycle Time % Breakdown by program
    timeBreakDownPerCycle_percent = px.bar(agg_prog, 
                                   x='time%', y='cycleID', 
                                   color='program', orientation='h', barmode='relative',
                                   labels={'time%':'% du Temps Total', 'program':'Programme', 'cycleID':'Cycles échantillonnés'}
                                   )
    timeBreakDownPerCycle_percent.update_yaxes(showticklabels=False)
    # plot(timeBreakDownPerCycle_percent)
    
    # Average Time per program accross all bars
    avgTimePerProgram = px.bar(agg_bars, x='program', y='avgProgTime',
                 color='progFreq',
                 labels={'program':'Programme','avgProgTime':'Temps moyen cumulé (ms)','progFreq':'Nombre <br>d\'apparitions<br>sur le stack ' }, height=1000)
    #plot(avgTimePerProgram)
    
    # Average Time per program - Residents only
    avgTimePerResidentProgram = px.bar(agg_residents, x='program', y='avgProgTime',
                 color='progFreq',
                 labels={'program':'Programme Résident','avgProgTime':'Temps (ms)', 'program':'Programme Résident', 'progFreq':'Nombre <br>d\'apparitions<br>sur le stack '}, height=1000)
    #plot(avgTimePerResidentProgram)  

# CHANGELOG - Replaced with csv reader from directory
#datetime.date(2022, 6, 13)
# changelog = np.array([['2022-06-13', '8:00AM', 'Modification du temps de lecture pour le laser robot de 1000ms à 220ms', 'PLC']])
# changelog_df = pd.DataFrame(changelog, columns=['Date', 'Heure', 'Descritpion', 'Localisation'])


# General Dashboard - streamlit ***

url_bc = 'https://beamcut.com/fr/'

# Sidebar ***
col1, mid, col2 = st.sidebar.columns([5,1,20])
with col1:
    st.image('BC_Avatar.jpg', width=60)
with col2:
    st.title('Beamcut Systems')
   
add_sidebar = st.sidebar.selectbox('Rapports', ('Stats Hebdomadaires', 'Stats Programmes BC', 'Changelog'))
st.sidebar.write("[Site Internet](%s)" % url_bc)

if add_sidebar == 'Stats Hebdomadaires':
    
    st.title('Étude de la production hebdomadaire')
    
    # Afficher les Périodes
    days_diff = numberOfDays - numberOfDays_prev
    
    
    st.subheader('Détail des Périodes')
    
    col1, col2 = st.columns(2)
    with col1:
        st.write('Période Courante')
        st.metric('Premier Jour', firstDay)
        st.metric('Dernier Jour', lastDay)
        st.metric('Échantillon', '{} Jours'.format(len(prodHourly)))
    with col2:
        st.write('Période Précédente')
        st.metric('Premier Jour', offset_firstDay)
        st.metric('Dernier Jour', offset_lastDay)
        st.metric('Échantillon', '{} Jours'.format(len(prodHourly_prev)))
   
    
    # Main Stats - Hourly Averages for the week
    
    # if Available data
    if prodDataAvailable:
        if prev_prodDataAvailable:
            st.metric('Nombre de Jours en production - Période Courante', len(prodDaily), delta=days_diff)
            st.metric('Moyenne Production Active Cumulée (HH:MM:SS) / Jour ', time.strftime('%H:%M:%S', time.gmtime(prodHourlyAgg['activeProd_s'])), s_to_hms(deltas.loc['activeProd_s'])) #delta=time.strftime('%H:%M:%S', time.gmtime(deltas.loc['activeProd_s'])) ) #time.strftime('%H:%M:%S', time.gmtime(prodHourlyAvg_thisWeek['activeProd_s'])
            st.metric('Moyenne de Barres / Heure',prodHourlyAgg['rawBars_h'], delta=round(deltas.loc['rawBars_h'],2))
            st.metric('Moyenne de Pieds Linéaires Brutes / Heure',prodHourlyAgg['handleRawFt_h'], delta=round(deltas.loc['handleRawFt_h'],2))
            st.metric('Moyenne de Pièces Produites / Heure',prodHourlyAgg['pieces_h'], delta=round(deltas.loc['pieces_h'],2))
            st.metric('Moyenne de Poids Usiné (lbs) / Heure',prodHourlyAgg['outWeightLbs_h'], delta=round(deltas.loc['outWeightLbs_h'],2))
            st.metric('Moyenne de Poids Usiné (tonnes) / Heure',prodHourlyAgg['outWeightTons_h'], delta=round(deltas.loc['outWeightTons_h'],2))
        else:
            # Current Metrics - No Deltas
            st.metric('Nombre de Jours en production - Période Courante', len(prodDaily))
            st.metric('Moyenne Production Active Cumulée (HH:MM:SS) / Jour ', time.strftime('%H:%M:%S', time.gmtime(prodHourlyAgg['activeProd_s']))) 
            st.metric('Moyenne de Barres / Heure',prodHourlyAgg['rawBars_h'])
            st.metric('Moyenne de Pieds Linéaires Brutes / Heure',prodHourlyAgg['handleRawFt_h'])
            st.metric('Moyenne de Pièces Produites / Heure',prodHourlyAgg['pieces_h'])
            st.metric('Moyenne de Poids Usiné (lbs) / Heure',prodHourlyAgg['outWeightLbs_h'])
            st.metric('Moyenne de Poids Usiné (tonnes) / Heure',prodHourlyAgg['outWeightTons_h'])
            st.markdown('**Les différentiels sont manquants car les données de production pour la période précédente sont non disponibles.<br /> **')
            
        # Detailed Stats - Hourly Averages per Day - current period
        # Formatted Dataframe
        st.subheader('Statistique à l\'heure pour chaque jour (période courante)')
        detailed_current = prodHourly[['prod_date', 'activeProd_t', 'rawBars_h', 'handleRawFt_h', 'pieces_h', 'outWeightLbs_h', 'outWeightTons_h']]
        detailed_current = detailed_current.round(decimals = 2)
        detailed_current.columns = ['Date', 'Temps de Production Total', 'Barres/h', 'Ft brutes/h', 'Pieces/h', 'Lbs/h', 'Tonnes/h']
        st.write(detailed_current)
        
        # Download Dataframe Option
        csv_current = convert_df(detailed_current)
    
        st.download_button(
           "Télécharger les Données du Tableau (CSV)",
           csv_current,
           "StatistiquesHorairesCourantes.csv",
           "text/csv",
           key='download-csv'
        )
        
        # Detailed Stats - Hourly Averages per Day - previous period
        # Formatted Dataframe
        st.subheader('Statistique à l\'heure pour chaque jour (courante et précédente)')
        detailed_previous = prodHourly_prev[['prod_date', 'activeProd_t', 'rawBars_h', 'handleRawFt_h', 'pieces_h', 'outWeightLbs_h', 'outWeightTons_h']]
        detailed_previous = detailed_previous.round(decimals = 2)
        detailed_previous.columns = ['Date', 'Temps de Production Total', 'Barres/h', 'Ft brutes/h', 'Pieces/h', 'Lbs/h', 'Tonnes/h']
        detailedAll = pd.concat([detailed_previous, detailed_current], keys=["precedente", "actuelle"])
        st.write(detailedAll)
        
        # Download Dataframe Option
        csv_all = convert_df(detailedAll)
    
        st.download_button(
           "Télécharger les Données du Tableau (CSV)",
           csv_all,
           "StatistiqueHoraire.csv",
           "text/csv",
           key='download-csv'
        )
    
    else:
        st.write(missingProdData_mess)
    
if add_sidebar == 'Stats Programmes BC':
    
    st.title('Étude des temps de programmes')
    # if program - process data available
    if processDataAvailable:
        st.subheader('Répartition du temps de cycle par programme exécuté')
        st.write('Un cycle représente une séquence de la cellule BC. Par exemple: Calibration, Renvoi du robot à \'Home\', Découpe de pièces dans une barre.')
        st.write(timeBreakDownPerCycle)
        st.write('Les programmes peuvent être retirés de la vue graphique en cliquant sur le nom dans la liste des programmes à droite.')
       
        st.subheader('Répartition du temps de cycle par programme exécuté en Pourcentage')
        st.write(timeBreakDownPerCycle_percent)
        st.write('Les programmes peuvent être retirés de la vue graphique en cliquant sur le nom dans la liste des programmes à droite.')
        
        st.subheader('Temps Moyen Cumulé pour une barre brute par chaque programme')
        st.write(avgTimePerProgram)
        # insert dataframe
        st.subheader('Temps Moyen Cumulé pour une barre brute par chaque programme résidents')
        st.write(avgTimePerResidentProgram)
        # insert dataframe
       
    else:
        st.write(missingProcessData_mess)
    
    
if add_sidebar == 'Changelog':
    st.title('Suivis des Changements')
    # 
    st.write(changelog)
    
    st.subheader('Ajouter un changement')
    
    #changeDate = st.date_input(label='Date du changement')
    changeDate = st.text_input(label='Date (aaaa-mm-jj)') # add regex ^\d{4}-\d{2}-\d{2}$
    changeTime = st.text_input(label='Heure de la mise à jour (ex: 8:00AM)') # add regex ^[\d{2}|\d{1}]:\d{2}[A|P]M$
    changeDesc = st.text_input(label='Description')
    changeLoc = st.text_input(label='Localisation (Ex: PLC, nom du programme, etc.')
    changeStatus = st.radio("Status de la modification", ('Ready', 'Testing', 'Retired', ))
    changeStatusDate = st.text_input(label='Date du status en cours (aaaa-mm-jj)') # add regex ^\d{4}-\d{2}-\d{2}$
    confirm = st.checkbox('Confirmer', value=False)

    if confirm:
        newlog = {'date':changeDate, 'time':changeTime, 'change':changeDesc, 'location':changeLoc, 'status':changeStatus, 'status_date': changeStatusDate}
        changelog = changelog.append(newlog, ignore_index=True)
        st.write('Nouveau Log:')
        st.write(changelog)
        changelog.to_csv('Changelog.csv', index=False)

        # upload to db

    # Option de modifier un changement

# onglet étude productivité de la shop: 
    # bridge avg time, cumulated downtime (sum of bridge times), cumulated active time
# Étude cumulative hebdomadaire (les nouvelles s'ajoutent pour un portrait global et cumulatif)

