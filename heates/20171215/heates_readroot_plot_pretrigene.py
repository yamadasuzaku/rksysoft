#!/usr/bin/env python

import sys, os, math, commands
import numpy as np
import matplotlib.pyplot as plt

params = {'xtick.labelsize': 13, # x ticks
          'ytick.labelsize': 13, # y ticks
          'legend.fontsize': 7
                    }

plt.rcParams.update(params)

#import pyfits

#from pyPdf import PdfFileWriter, PdfFileReader

argvs = sys.argv
print argvs

if len(argvs) < 2:
     print 'usage : %s rootfile chan'%os.path.basename(argvs[0])
     quit()

rootfile=argvs[1]
channel=int(argvs[2])

print " rootfile = ", rootfile
print " channel  = ", channel

import ROOT

#quit()

#### variable
rootfiletag = rootfile.replace(".root","")
#rootfiletag = 'run0002_0000_mass_BB'
#rootfiletag  = 'run0004_0000_mass_BC'
rootfilename =  rootfiletag + '.root'
#channel = 1
plotpulse = False
valgood = 1
#### const
LEN_PULSE_RECORD = 1024

#### global settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['xtick.labelsize'] = 'small'
plt.rcParams['ytick.labelsize'] = 'small'
plt.rcParams['figure.subplot.left'] = 0.15
plt.rcParams['figure.subplot.right'] = 0.95
plt.rcParams['figure.subplot.bottom'] = 0.10
plt.rcParams['figure.subplot.top'] = 0.90
plt.rcParams['figure.subplot.wspace'] = 0.2
plt.rcParams['figure.subplot.hspace'] = 0.2
#plt.rcParams['axes.formatter.limits'] = -3, 3

plt_font = {
    'family' : 'serif',
    'color'  : 'black',
    'weight' : 'normal',
    'size'   : 16,
}

#### functions
def common_hash(rootfile):
    """
    ['ch', 'col', 'row', 'ncols', 'nrows', 'npulses', 'nsample',
     'npresamples', 'row_timebase', 'timebase', 'timestamp_offset']
    """

    tree = rootfile.Get('common')
    namelst = [b.GetName() for b in tree.GetListOfBranches()]

    tbl = {}
    nct = tree.GetEntries()
    for i in xrange(nct):
        tree.GetEntry(i)
        tmp = {}
        for n in namelst:
            if n == 'ch':
                pass
            else:
                tmp[n]=tree.__getattr__(n)

        tbl[tree.__getattr__('ch')]=tmp

    return tbl


def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=numpy.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]

    #print(len(s))
    if window == 'flat': #moving average
        w=numpy.ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')

    y=numpy.convolve(w/w.sum(),s,mode='valid')
#    y=numpy.convolve(w/w.sum(),s,mode='same')
    return y



def create_image(pulse_record, imgtitle='', pdfoutput=False, pdfname='img.pdf', params = ['']):


    fig = plt.figure()
    p0 = fig.add_subplot(111)

    # p0.axis([xmin,xmax,ymin,ymax])
    p0.grid(b=True, which='major')

    p0.set_title(imgtitle, fontdict=plt_font)

    p0.set_xlabel('time', fontdict=plt_font)
    p0.xaxis.set_label_coords(0.5, -0.05)
    p0.set_xscale('linear', noposx='clip')

    p0.set_ylabel('adc', fontdict=plt_font)
    p0.yaxis.set_label_coords(-0.1, 0.5)
    p0.set_yscale('linear', noposy='clip')

    p0.errorbar(range(LEN_PULSE_RECORD), pulse_record, linestyle='-', label = "no filter")

    gap = 10
#    gap = 5
    fnum = 2 * gap + 1


    filt_pulse_record = smooth(pulse_record, fnum, 'hanning')

    filt_pulse_record = filt_pulse_record[gap: -1*gap] 

#    print " len(pulse_record)  = ", len(pulse_record)
#    print " len(filt_pulse_record)  = ", len(filt_pulse_record)
    
    p0.errorbar(range(LEN_PULSE_RECORD), filt_pulse_record, linestyle='-', label = "filter")
    plt.legend(numpoints=1, frameon=False, loc="best")

#    p0.errorbar(range(LEN_PULSE_RECORD), pulse_record, marker='.', linestyle='')
    plt.figtext(0.1,0.93, str(params))

    if pdfoutput:
        fig.set_size_inches(12.0,9.0)
        plt.savefig(pdfname, format='pdf', dpi=1000)
    else:
        if plotpulse:
            plt.show()
    plt.close()

    # plot diff 

    fig = plt.figure()
    p0 = fig.add_subplot(111)

    # p0.axis([xmin,xmax,ymin,ymax])
    p0.grid(b=True, which='major')

    p0.set_title(imgtitle, fontdict=plt_font)

    p0.set_xlabel('time', fontdict=plt_font)
    p0.xaxis.set_label_coords(0.5, -0.05)
    p0.set_xscale('linear', noposx='clip')

    p0.set_ylabel('adc', fontdict=plt_font)
    p0.yaxis.set_label_coords(-0.1, 0.5)
    p0.set_yscale('linear', noposy='clip')

    diff = pulse_record - filt_pulse_record 
    p0.errorbar(range(LEN_PULSE_RECORD), diff, linestyle='-', label = "diff")
    plt.legend(numpoints=1, frameon=False, loc="best")

    p0.set_ylim(-200.,200.)

#    p0.errorbar(range(LEN_PULSE_RECORD), pulse_record, marker='.', linestyle='')
    plt.figtext(0.1,0.93, str(params))

    if pdfoutput:
        fig.set_size_inches(12.0,9.0)
        plt.savefig(pdfname, format='pdf', dpi=1000)

    else:
        if plotpulse:
            plt.show()
    plt.close()



def write_fits(x, fname = "test.fits"):

    outfname = fname.replace(".fits","_tmp.fits")
    hdu = pyfits.PrimaryHDU(x)
    hdu.writeto(outfname, clobber=True)


def fourplot(x1,y1,y2, outfname = "plot_pretrigene.pdf", tag="energy", autoscale = False, de = 200.):
     # x1 = time
     # y1 = pretrig_mean
     # y2 = energy 
     
     y1mean = np.mean(y1)

     F = plt.figure(figsize=(12,8))

     # time vs. pretrig_mean - mean
     ax = plt.subplot(2,2,1)     
#     plt.title(rootfilename + " ch = " + str(channel)) 
     plt.figtext(0.2,0.94, rootfilename + " ch = " + str(channel) + " " + tag, size='x-small') 
     plt.ylabel('pretrig_mean - \n premean(' +str("%4.2f" % y1mean) + ")" )
     plt.xlabel('Time')    
     plt.grid(True)
     plt.xscale('linear')
     plt.yscale('linear')
     ax.set_ylim(-50, 50)     
     plt.errorbar(x1, y1 - y1mean, fmt='ro', label="CH = " + str(channel), ms=2)    
     plt.legend(numpoints=1, frameon=False, loc='best')

     # time vs. energy 
     ax = plt.subplot(2,2,2)     
     plt.ylabel(tag)

     if autoscale:
          ax.set_ylim(np.median(y2) - de, np.median(y2) + de)
     else:
          ax.set_ylim(5800, 6000)     

     plt.xlabel('Time')    
     plt.grid(True)
     plt.xscale('linear')
     plt.yscale('linear')
     plt.errorbar(x1, y2, fmt='ro', label="CH = " + str(channel), ms=2)    
     plt.legend(numpoints=1, frameon=False, loc='best')


     # pretrig_mean - mean vs. energy 
     ax = plt.subplot(2,2,3)     
     plt.ylabel(tag)
     plt.xlabel('pretrig_mean - premean(' +str(y1mean) + ")" )
     plt.grid(True)

     if autoscale:
          ax.set_ylim(np.median(y2) - de, np.median(y2) + de)
     else:
          ax.set_ylim(5800, 6000)     

     ax.set_xlim(-50, 50)     
     plt.xscale('linear')
     plt.yscale('linear')
     plt.errorbar(y1 - y1mean, y2, fmt='ro', label="CH = " + str(channel), ms=2)    
     plt.legend(numpoints=1, frameon=False, loc='best')


     # pretrig_mean - mean vs. energy 
     ax = plt.subplot(2,2,4)     
     plt.ylabel("counts / 2eV")     
     plt.xlabel(tag)
     plt.grid(True)
     plt.xscale('linear')
     plt.yscale('linear')

     if autoscale:
          histy, histx, pacthes = plt.hist(y2, color='b', bins = 101, range = (np.median(y2) - de, np.median(y2) + de), label="CH = " + str(channel))
     else:
          histy, histx, pacthes = plt.hist(y2, color='b', bins = 101, range = (5800,6000), label="CH = " + str(channel))

     plt.legend(numpoints=1, frameon=False, loc='best')
     
     plt.savefig(outfname)


#### MAIN
if __name__ == '__main__':


    allpulse = []

    rootfile=ROOT.TFile(rootfilename)
    # common_hash = common_hash(rootfile)
    # cmn=common_hash[channel]

    # outpdf=PdfFileWriter()

    pulsetreename='chan%d'%(channel)

    try:
         pulsetree=rootfile.Get(pulsetreename)    
         npt=pulsetree.GetEntries()
    except:
         print "No data found."
         sys.exit()

    sum_pulse = np.zeros(LEN_PULSE_RECORD)

    nump = 0
    tinit = 0. 

    pmean = []
    time = []
    ene = []
    fval = []

    for i in xrange(npt):
        pulsetree.GetEntry(i)

#        ev             = pulsetree.ev
#        good           = pulsetree.good
#        filt_phase     = pulsetree.filt_phase
        filt_value     = pulsetree.filt_value
#        filt_value_dc  = pulsetree.filt_value_dc
#        filt_value_phc = pulsetree.filt_value_phc
#        filt_value_tdc = pulsetree.filt_value_tdc
#        min_value      = pulsetree.min_value
#        peak_index     = pulsetree.peak_index
#        peak_time      = pulsetree.peak_time
#        peak_value     = pulsetree.peak_value
#        postpeak_deriv = pulsetree.postpeak_deriv
        pretrig_mean   = pulsetree.pretrig_mean
#        pretrig_rms    = pulsetree.pretrig_rms
#        promptness     = pulsetree.promptness
#        pulse_average  = pulsetree.pulse_average
#        pulse_rms      = pulsetree.pulse_rms
#        rise_time      = pulsetree.rise_time
        timestamp      = pulsetree.timestamp
#        pulse_record   = np.array(pulsetree.pulse_record)
        energy   = pulsetree.energy
        
        if i == 0:
             tinit = timestamp 

        time.append(timestamp - tinit)
        pmean.append(pretrig_mean)
        ene.append(energy)
        fval.append(filt_value)
        
    time = np.array(time)             
    pmean = np.array(pmean)
    ene = np.array(ene)
    fval = np.array(fval)

    outputfiguredir = "figures_plot_pretrigene"
    commands.getoutput('mkdir -p ' + outputfiguredir)

    outfname = outputfiguredir + "/" + rootfiletag + "_pulse_chan" + str("%03d" % channel) + ".pdf"
    fourplot(time, pmean, ene, outfname = outfname, autoscale = False)

    outfname = outputfiguredir + "/" + rootfiletag + "_pulse_chan" + str("%03d" % channel) + "_filt_value.pdf"
    fourplot(time, pmean, fval, outfname = outfname, tag = "filt_value", autoscale = True)


    print " time  = from ", time[0], " to ", time[-1]
    print " pmean = from ", np.min(pmean), " to ", np.amax(pmean), " std = ", np.std(pmean)
    print " ene = from ", np.min(ene), " to ", np.amax(ene), " std = ", np.std(ene)
    print " fval = from ", np.min(fval), " to ", np.amax(fval), " std = ", np.std(fval)
