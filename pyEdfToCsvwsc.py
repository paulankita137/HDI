#!/usr/bin/python3
import pyedflib
import numpy as np
import argparse
import datetime
import locale

import numba

import os
import sys
GPUID= sys.argv[0]

#os.environ["CUDA_VISIBLE_DEVICES"]=GPUID
from numba import jit




# defaults
separator = ';'
locale.setlocale(locale.LC_ALL, '')
decimalpoint = (locale.localeconv()['decimal_point'])
@jit

def signalsToCsv(filename, labels, signals):
    ''' Export all signals to single .csv'''
    filename = filename+'.csv'
    with open(filename, 'w+') as f:
        labels_row = separator.join(labels)
        f.write(labels_row)

        # Get max samples length
        maxLength = 0
        for i in range(len(signals)):
            maxLength = max(maxLength, len(signals[i]))

        # @TODO


def signalsToCsvs(filename, labels, signals, sampleRates, dimensions):
    ''' Export all signals to multiple .csv'''
   # for i in range(len(signals)):
    i=15
    j=0
    filepath = filename+labels[i]+'_'+str(sampleRates[i])+'sps.csv'
    with open(filepath, 'w+') as f:
            # Labels
            f.write('Time[s]%c%s [%s]\n' %
                    (separator, labels[i], dimensions[j]))

            # Prepare time values
            if (args.timeAbsolute):
                time = startTime
                delta = datetime.timedelta(seconds=1.0/sampleRates[i])
            else:
                time = 0
                delta = 1.0/sampleRates[i]

            # Samples saving
            for sample in signals[j]:
                # DateTime to text
                if (args.timeAbsolute):
                    # Absolute time used
                    text = '%s%c' % (time.strftime(
                        '%Y-%m-%d %H:%M:%S.%f'), separator)
                else:
                    # Relative time used
                    text = '%2.4f%c' % (time, separator)
                time += delta
                # Sample to text
                text += str(sample)
                # EOL
                text += '\n'
                # Decimal mark conversion of whole line
                if (decimalpoint == ','):
                    text = text.replace('.', ',')
                # Save
                f.write(text)


# Arguments and config
# #####################################################
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str,
                    required=True, help='Input EDF file')
parser.add_argument('-s', '--separator', type=str,
                    required=False, help='Data CSV separator')
parser.add_argument('-d', '--decimalpoint', type=str,
                    required=False, help='Decimal point character')
parser.add_argument('-t', '--timeAbsolute', action='store_true',
                    required=False, help='Default behaviour is relative time printing (First sample is 0s). Absolute time prints time according to record start time.')
args = parser.parse_args()

if (args.separator is not None):
    separator = args.separator

if (args.decimalpoint is not None):
    decimalpoint = args.decimalpoint


# Open EDF file
f = pyedflib.EdfReader(args.input)
print('(File) Opened', args.input)
n = f.signals_in_file
print('(File) %u signals in file.' % (n))
labels = f.getSignalLabels()
print('(File) Signal labels in file : ', labels)
startTime = f.getStartdatetime()
print('(File) Start of recording', startTime)

sampleRates = f.getSampleFrequencies()
signals = []
dimensions = []
print('labels15',labels[15])

#for i in np.arange(n):
#    print('(Signal) Reading signal %u `%s`, sampling freqeuncy %u, samples %u' % (
#        i, labels[i], f.getSampleFrequency(i), f.getNSamples()[i]))
#    print('(Signal) Signal header : ', f.getSignalHeader(i))
#    print('')
#    signal = f.readSignal(i)
#    signals.append(signal)
#    dimensions.append(f.getPhysicalDimension(i))



i=15
if i == 15:
    print('(Signal) Reading signal %u `%s`, sampling freqeuncy %u, samples %u' % (
            i, labels[i], f.getSampleFrequency(i), f.getNSamples()[i]))
    print('(Signal) Signal header : ', f.getSignalHeader(i))
    print('')
    signal = f.readSignal(i)
    signals.append(signal)
    dimensions.append(f.getPhysicalDimension(i))
    print('Creation of .csv.')
    signalsToCsvs(args.input, labels, signals, sampleRates, dimensions)
# Create .csv
#print('Creation of .csv.')
#signalsToCsvs(args.input, labels, signals, sampleRates, dimensions)


#delete non SaO2 files

import glob,os

for f in glob.glob("*ECG*.csv"):
    os.remove(f)
for f in glob.glob("*EMG*.csv"):
    os.remove(f)

for f in glob.glob("*EOG*.csv"):
    os.remove(f)

#for f in glob.glob("*H.R*.csv"):
 #   os.remove(f)
for f in glob.glob("*ABD*.csv"):
    os.remove(f)

for f in glob.glob("*AIRF*.csv"):
    os.remove(f)

for f in glob.glob("*LIGH*.csv"):
    os.remove(f)

for f in glob.glob("*OX*.csv"):
    os.remove(f)

for f in glob.glob("*POSI*.csv"):
    os.remove(f)

for f in glob.glob("*THO*.csv"):
    os.remove(f)

for f in glob.glob("*NEW*.csv"):
    os.remove(f)

for f in glob.glob("*SOU*.csv"):
    os.remove(f)

for f in glob.glob("*EEG*.csv"):
    os.remove(f)

for f in glob.glob("*AUX*.csv"):
    os.remove(f)

for f in glob.glob("*H.R.*.csv"):
    os.remove(f)


###Removals for WSC

for f in glob.glob("*abdomen*.csv"):
    os.remove(f)

for f in glob.glob("*edfC3*.csv"):
    os.remove(f)

for f in glob.glob("*edfE1*.csv"):
    os.remove(f)

for f in glob.glob("*chin*.csv"):
    os.remove(f)

for f in glob.glob("*edfE2*.csv"):
    os.remove(f)

for f in glob.glob("*leg*.csv"):
    os.remove(f)

for f in glob.glob("*nas*.csv"):
    os.remove(f)

for f in glob.glob("*edfO2*.csv"):
    os.remove(f)

for f in glob.glob("*oral*.csv"):
    os.remove(f)

for f in glob.glob("*position*.csv"):
    os.remove(f)

for f in glob.glob("*O1.*.csv"):
    os.remove(f)
for f in glob.glob("*sum.*.csv"):
    os.remove(f)



#numba.cuda.profile_stop()    

