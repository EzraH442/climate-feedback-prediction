 ################################################
#    Script to test the global neural network    #
#                 By: Diana Diaz                 #
 ################################################

## ML libraries
# tensorflow
import tensorflow as tf
from tensorflow import keras

## import general libraries 
import os,sys,glob
import numpy as np
import xarray as xr

## import libraries for plots
import matplotlib.pyplot as plt
import matplotlib as mpl
p
import pandas as pd

from mpl_toolkits.basemap import Basemap

import matplotlib as mpl
from scipy.stats import linregress
# Set the font size globally
font_size = 20
mpl.rcParams['font.size'] = font_size

global_nn_model/nn_train/train6/nn_saved_model'
path_saved_model = '../data/nn_saved_model'
model = keras.models.load_model(path_saved_model)


## import data
# single levels data including effective cloud optical depth variable
path = '../data/'
file_sl = 'era5_1deg_monthly_avg_sl_1990-2020_tisr_tciw_tclw_tcwv_hcc_mcc_lcc_sp_tco3_fal_tsr_msl_ecod_noZeros.nc'
xrr_sl = xr.open_dataset(path+file_sl)

#To convert to W m-2 , the accumulated values should be divided by the accumulation period expressed in seconds
xrr_sl['tisr'] = xrr_sl.tisr/24/3600
xrr_sl['tsr'] =xrr_sl.tsr/24/3600

#xrr_sl['tsr'] = xr.where(xrr_sl.tsr < 0, np.nan, xrr_sl.tsr)
#xrr_sl['tisr'] = xr.where(xrr_sl.tisr < 0, np.nan, xrr_sl.tisr)

## divide datasets for training and testing 
# extract the years from the time dimension
years = xrr_sl['time.year']

# select odd years
#odd years for testing
xrr_sl_odd = xrr_sl.sel(time=(years % 2 == 1))

# select even years
#even years for trainning
xrr_sl_even = xrr_sl.sel(time=(years % 2 == 0))

## We test the NN with the odd years
df_test = xrr_sl_odd.to_dataframe()
vars_list_test =  [ np.array(df_test.tisr), np.array(df_test.tciw), np.array(df_test.tclw), np.array(df_test.tcwv),                   
                   np.array(df_test.lcc), np.array(df_test.mcc), np.array(df_test.hcc), np.array(df_test.sp), 
                   np.array(df_test.tco3), np.array(df_test.fal),np.array(df_test.ecod),
                   np.array(df_test.ecod*df_test.fal)]

tsr_test = np.array(df_test.tsr)

max_val_test = np.round(np.max(tsr_test),2)
min_val_test = np.round(np.min(tsr_test),2)

# Normalization range [a,b]
#We set the normalization rate to be between -1 and 1, since the activation function (tanh) has the same range
a = -1
b = 1
nrmlzd_vars_list_test = []
for x in vars_list_test:    
    n = MinMaxNorm(x,a,b)
    nrmlzd_vars_list_test.append(n)

#we combine the input variables into one merged array
merged_array_test = np.stack((nrmlzd_vars_list_test), axis=1) #da.stack((nrmlzd_vars_list), axis=1)
xx_test = merged_array_test
yy_test = MinMaxNorm(tsr_test,a,b) #we normalize the output

#predictions
prdctns=model.predict(xx_test)
#inverse transform
prdctnss = MinMaxInverse(prdctns,a,b,df_test.tsr)
#df_lat_lon = df_test.reset_index(level = ['lon','lat'])
#df_lat_lon['NN_prdctns'] = prdctnss
#df_lat_lon.to_csv('dataset_nn_test.csv')

df_test['NN_prdctns'] = prdctnss

# Convert DataFrame to xarray Dataset
dataset = xr.Dataset.from_dataframe(df_test.squeeze())

tsr_mean = dataset['tsr'].mean(dim='time')
tsr_mean = tsr_mean.transpose()
TSR = np.round(np.mean(tsr_mean),2)
TSR_str = str(TSR.values)
TSR_plot = TSR_str

nnPred_mean = dataset['NN_prdctns'].mean(dim='time')
nnPred_mean = nnPred_mean.transpose()
NN_TSR = np.round(np.mean(nnPred_mean),2)
NN_TSR_str = str(NN_TSR.values)
NN_TSR_plot = NN_TSR_str

if np.max(tsr_mean)> np.max(nnPred_mean):
    max_val = np.max(tsr_mean)
else:
    max_val = np.max(nnPred_mean)

fig = plt.figure(figsize=(8, 6),dpi=300)

m = Basemap(projection='cyl', resolution='l',
            llcrnrlat=-90, urcrnrlat=90,
            llcrnrlon=0, urcrnrlon=360)


lon, lat = dataset['lon'].values, dataset['lat'].values
lon, lat = np.meshgrid(lon, lat)
x, y = m(lon, lat)
m.pcolormesh(x, y, tsr_mean, shading='nearest', cmap='Spectral',vmin=0,vmax=max_val)

m.drawcoastlines()
m.drawcountries()
m.drawmapboundary()

# Draw parallels (latitude lines) and meridians (longitude lines) with labels
parallels = np.arange(-90., 91., 30.)  # Define the latitude lines you want to draw
meridians = np.arange(-180., 181., 60.)  # Define the longitude lines you want to draw
m.drawparallels(parallels, labels=[True, False, False, True])  # Draw latitude lines with labels on left and right
m.drawmeridians(meridians, labels=[True, False, False, True])  # Draw longitude lines with labels on top and bottom

plt.text(np.max(tsr_mean)-65, np.max(dataset['lat'].values)+5 , TSR_plot, fontsize=20)
plt.colorbar(orientation='horizontal', fraction=0.075, label='$W/m^2$')
plt.title('TSR (ERA5)')

#plt.show()
plt.savefig('tsr_test.png')

fig = plt.figure(figsize=(8, 6),dpi=300)

m = Basemap(projection='cyl', resolution='l',
            llcrnrlat=-90, urcrnrlat=90,
            llcrnrlon=0, urcrnrlon=360)


lon, lat = dataset['lon'].values, dataset['lat'].values
lon, lat = np.meshgrid(lon, lat)
x, y = m(lon, lat)
m.pcolormesh(x, y, nnPred_mean, shading='nearest', cmap='Spectral', vmin =0,vmax= max_val ) # ,vmin= -120, vmax=40 )

m.drawcoastlines()
m.drawcountries()
m.drawmapboundary()

# Draw parallels (latitude lines) and meridians (longitude lines) with labels
parallels = np.arange(-90., 91., 30.)  # Define the latitude lines you want to draw
meridians = np.arange(-180., 181., 60.)  # Define the longitude lines you want to draw
m.drawparallels(parallels, labels=[True, False, False, True])  # Draw latitude lines with labels on left and right
m.drawmeridians(meridians, labels=[True, False, False, True])  # Draw longitude lines with labels on top and bottom

plt.text(np.max(nnPred_mean)-65, np.max(dataset['lat'].values)+5 , NN_TSR_plot, fontsize=20)
plt.colorbar(orientation='horizontal', fraction=0.075, label='$W/m^2$')

plt.title('TSR (NN)')

#plt.show()
plt.savefig('nn_prdctns.png')



difference = nnPred_mean - tsr_mean

if abs(np.min(difference))>abs(np.max(difference)):
    value = abs(np.min(difference))
else:
    value = abs(np.max(difference))
    
MBE = np.round(np.mean(difference),2)
MBE_str = str(MBE.values)
MBE_plot = MBE_str
    

fig = plt.figure(figsize=(8, 6),dpi=300)

m = Basemap(projection='cyl', resolution='l',
            llcrnrlat=-90, urcrnrlat=90,
            llcrnrlon=0, urcrnrlon=360)

lon, lat = dataset['lon'].values, dataset['lat'].values
lon, lat = np.meshgrid(lon, lat)
x, y = m(lon, lat)
m.pcolormesh(x, y, difference, shading='nearest', cmap='bwr' ,vmin =-value,vmax=value)

m.drawcoastlines()
m.drawcountries()
m.drawmapboundary()

# Draw parallels (latitude lines) and meridians (longitude lines) with labels
parallels = np.arange(-90., 91., 30.)  # Define the latitude lines you want to draw
meridians = np.arange(-180., 181., 60.)  # Define the longitude lines you want to draw
m.drawparallels(parallels, labels=[True, False, False, True])  # Draw latitude lines with labels on left and right
m.drawmeridians(meridians, labels=[True, False, False, True])  # Draw longitude lines with labels on top and bottom

plt.text(320, np.max(dataset['lat'].values)+5 , MBE_plot, fontsize=20)
plt.colorbar(orientation='horizontal', fraction=0.075, label='$W/m^2$')

plt.title('MBE')

#plt.show()
plt.savefig('mbe.png')


rmse_plot = np.sqrt((difference)**2)

RMSE = np.round(np.sqrt(np.mean((difference)**2)),2)
RMSE_str = str(RMSE.values)
RMSE_plot = RMSE_str

fig = plt.figure(figsize=(8, 6),dpi=300)

m = Basemap(projection='cyl', resolution='l',
            llcrnrlat=-90, urcrnrlat=90,
            llcrnrlon=0, urcrnrlon=360)


lon, lat = dataset['lon'].values, dataset['lat'].values
lon, lat = np.meshgrid(lon, lat)
x, y = m(lon, lat)
m.pcolormesh(x, y, rmse_plot, shading='nearest', cmap='Blues' ,vmin =0,vmax=np.max(rmse_plot))

m.drawcoastlines()
m.drawcountries()
m.drawmapboundary()

# Draw parallels (latitude lines) and meridians (longitude lines) with labels
parallels = np.arange(-90., 91., 30.)  # Define the latitude lines you want to draw
meridians = np.arange(-180., 181., 60.)  # Define the longitude lines you want to draw
m.drawparallels(parallels, labels=[True, False, False, True])  # Draw latitude lines with labels on left and right
m.drawmeridians(meridians, labels=[True, False, False, True])  # Draw longitude lines with labels on top and bottom

plt.text(325, np.max(dataset['lat'].values)+5 , RMSE_plot, fontsize=20)
plt.colorbar(orientation='horizontal', fraction=0.075, label='$W/m^2$')

plt.title('RMSE')

#plt.show()
plt.savefig('rmse.png')


slope, intercept, r_value, p_value, std_err = linregress(prdctnss.squeeze(), tsr_test)

r2 = r_value**2
r_squared= np.round(r2,3)
r_squared_str = str(r_squared)
r_squared_plot = r_squared_str+' $W/m^2$'

mse = np.mean((tsr_test-prdctnss.squeeze())**2)
rmse = np.sqrt(mse)

val_str = '$R^2$ = '+ r_squared_plot+' \nRMSE = ' + RMSE_plot

mpl.rcParams['font.size'] = 12

plt.figure(figsize=(6, 6), dpi=300) #80

# Create a diagonal line royalblue
plt.plot(prdctnss.squeeze(), np.array(prdctnss.squeeze()) * slope + intercept, color='k')
plt.scatter(prdctnss,tsr_test,color='royalblue', marker = '.',s=0.1)

plt.ylabel('TSR from ERA5 $[W/m^2$]')
plt.xlabel('TSR predicted by the NN [$W/m^2$]')
plt.title('Validation of the climatological TSR')
plt.text(0, 370, val_str, fontsize=12)
plt.savefig('validation.png')

# saving metrics in txt file
file_txt = "metrics.txt"
file = open(file_txt, "w")

file.write('MBE = {}\n'.format(MBE_str))
file.write('RMSE = {}\n'.format(RMSE_str))
file.write('TSR = {}\n'.format(TSR_str))
file.write('NN_TSR = {}\n'.format(NN_TSR_str))

file.close()
