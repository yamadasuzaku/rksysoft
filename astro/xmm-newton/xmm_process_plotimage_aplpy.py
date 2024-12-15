# start witm ipython --pylab

import aplpy
import numpy as np 
import matplotlib.pyplot as plt

params = {'xtick.labelsize': 13, # x ticks
          'ytick.labelsize': 13, # y ticks
          'legend.fontsize': 10
                    }
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update(params)

ximg1 = "sumcor_adapt-400-750-all.fits"
ximg2 = "sumcor_adapt-750-1300-all.fits"
ximg3 = "sumcor_adapt-2000-7200-all.fits"
radio1 = "VLA-remake_4_TAN_J2000.fits"

# x-ray bright region 
ra = 288.55  # deg
dec = 4.94   # deg
width = 1.7  # deg 
height = 0.8 # deg

vmin = 1e-1
vmax = 300
contnum = 15

inputimg = [ximg1, ximg2, ximg3, radio1]

for i, file in enumerate(inputimg):

    fig = plt.figure(figsize=(15, 8))

    print "start ....", file 
    outfname = file.replace(".fits","_aplpy.png")

    figapl = aplpy.FITSFigure(file,figure=fig)
    figapl.set_title(file)

    #figapl.set_theme('publication')
    figapl.set_theme('pretty')

    if i < 3: # XMM
        figapl.show_colorscale(vmin=vmin,vmax=vmax, stretch='log', cmap='jet')
    else : # VLA
        tvmin = 1e-3
        tvmax = 0.10
        figapl.show_colorscale(vmin=tvmin,vmax=tvmax, stretch='log', cmap='jet')

    figapl.recenter(ra, dec, width=width, height=height)  # degrees
    figapl.show_contour(radio1, colors='black', levels=np.logspace(-3, -1., contnum), linewidths = 1, alpha = 0.4)
    figapl.add_colorbar()
    figapl.colorbar.set_width(0.3)
    figapl.colorbar.set_location('right')
    figapl.add_grid()
    figapl.grid.set_color('black')
    figapl.grid.set_alpha(0.2)

		
    figapl.add_scalebar(0.05275, "0.05275 degrees", color='white', corner='top right') # assuming ~ 4.2 kpc 
    figapl.scalebar.set_label("4 pc")
    
    figapl.tick_labels.set_xformat('hh:mm:ss')
    figapl.tick_labels.set_yformat('dd:mm:ss')
    
    figapl.save(outfname)

    fig.clf()
