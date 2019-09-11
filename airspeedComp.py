import numpy as np
import pandas as pd
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import glob
import scipy

def getDataFaam(allFlights = True):
	"""
	A function that gets all the data in the variables list from all data on a USB (Flights C124-C198)
	and appends it to a single pandas dataframe

	outputs:
	pandas dataframe
	"""
	variables = ["Time", "PS_RVSM", "Q_RVSM", "ROLL_GIN", "ALT_GIN", "TAS_RVSM","TAS","PSP_TURB_FLAG"]
	dfFlights = pd.DataFrame(columns = variables)
	if allFlights == True:
		#Get all data (C124 - C198)
		fnames = sorted(glob.glob("/media/deckel/417D-0C9F/core_data2/*.nc"), reverse = True)		
	for n, i in enumerate(fnames):
		if n > 25:
			break
		print(i)
		fh = Dataset(i,'r')
		df = pd.DataFrame(columns = variables)
		for j in variables:
			var = fh.variables[j][:]
			var = var.ravel()
			df[j] = var
		df = df.rename(columns = {"Time":"TIME"})
		df["flightNumber"] = n
		dfFlights = dfFlights.append(df)
	return(dfFlights, fnames)

def straightRuns(df):
	"""
	A function that returns only straight and level runs.
	Removes all data that dosn't meet the following requirements:
	Altitude higher that 1500m
	Roll angle no bigger than += 8 degrees
	The STD of the altitude over 2 seconds is no more that 1m
	"""
	df = df[df["ALT_GIN"] > 1500]
	df = df[np.array((df["ROLL_GIN"] <= 8) & ((df["ROLL_GIN"] >= -8)))]
	df = df[df["ALT_GIN"].rolling(64).std() < 1]
	return(df)

def sp_mach(psp, sp):
	"""
	Return mach number
	
	inputs:
	Pitot-static pressure mb
	static pressure mb

	outputs:
	mach number
	"""
	mach = np.sqrt(5.0 * ((1.0 + psp / sp)**(2.0 / 7.0) - 1.0))
	return(mach)

def regression(dataX, dataY, order):
	"""
	Get regression functions
	"""
	fit = np.polyfit(dataX, dataY, order)
	fitFn = np.poly1d(fit)

	# R squared
	yhat = fitFn(sorted(dataX))
	ybar = np.sum(sorted(dataY))/len(dataY)
	sstol = np.sum((sorted(dataY)-ybar)**2)
	ssreg = np.sum((yhat-ybar)**2)
	r = ssreg/sstol

	return(fitFn, fit,r)

# Get data
df, fnames = getDataFaam()
# Remove all btu straight and level runs from data
df = straightRuns(df)
# Remove bad data points when probe was iced
df = df[df["PSP_TURB_FLAG"] == 0.0]
# Calculate mach number
df["MACH"] = sp_mach(df["Q_RVSM"], df["PS_RVSM"])
# Get flight numbers
firstflight = fnames[25][62:66]
lastFlight = fnames[0][62:66]
# Get fit function
fitFn, fit,r = regression(df["MACH"], df["TAS_RVSM"]-df["TAS"], 2)
# Get 2004 data to show drift
oldData = pd.read_csv("/media/deckel/417D-0C9F/dataset.csv")
# Plot Data
plt.scatter(df["MACH"], df["TAS_RVSM"]-df["TAS"], s=0.5, c = "b",label="C173-198 Level Runs", zorder=10)
plt.scatter(oldData["mach"], oldData["error"] ,marker= "+", c="black", label="B001-012 Level Runs", zorder=11)
plt.plot(sorted(df["MACH"].values), fitFn(sorted(df["MACH"].values)), c="black", label="C173-198 Level Runs Best fit", zorder=12)
plt.title("Difference in TAS from {} - {} level runs compared to measurements from 2004".format(firstflight,lastFlight))
plt.xlabel("Mach number")
plt.ylabel("RVSM TAS - P0_TAS / ms-1")
plt.text(0.33,4,r"$y = {:.2f}x^2 + {:.2f}x +{:.2f}$".format(fit[0],fit[1],fit[2]))
plt.text(0.33,3.6,r"$r^2 = {:.3}$".format(r))
plt.savefig("AirSpeedComparisonTransparent.png", dpi = 1000)
plt.grid(True, zorder=0)
plt.legend(loc="upper left")
plt.show()