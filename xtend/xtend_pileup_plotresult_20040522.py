import pandas as pd
import matplotlib.pyplot as plt
params = {'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 8}
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update(params)
from io import StringIO

# CSV data
data = """OBSID,pileup fraction 3 percent,pileup fraction 1 percent,name,mode,exposure,rate
000125000,6.315789473684211,14.736842105263158,NGC4151,WINDOW2,33054.94270685315,5.7093428257817855
000125000,6.315789473684211,10.526315789473685,NGC4151,WINDOW2BURST,4118.068079590797,6.150699675299414
000127000,10.526315789473685,18.94736842105263,PKS2155-304,WINDOW2,84364.8549336791,11.267357725527546
000137000,6.315789473684211,18.94736842105263,NGC4151,WINDOW2,54083.16966277361,8.73898484402849
000139000,27.36842105263158,40.0,Vela_X-1_1,WINDOW2BURST,1900.799580156803,56.70403188489164
000145000,6.315789473684211,14.736842105263158,3C273,WINDOW2,86690.72352400422,6.001022702918728
000161000,-1,10.526315789473685,MCG-6-30-15,WINDOW1,49.55763185024261,15.961214659939964
000161000,6.315789473684211,14.736842105263158,MCG-6-30-15,WINDOW2,114666.9159982502,6.884060612670927
000162000,-1,-1,Circinus_Galaxy,WINDOW2,286799.5506817997,1.0870832930484966
100001010,-1,-1,G21.5-0.9,WINDOW1,57929.1955935955,11.225928365418154
300018010,-1,-1,M81,WINDOW1,239292.322055608,12.20583249353591
300028010,10.526315789473685,23.157894736842103,Circinus_X-1,WINDOW2,26215.15315827727,14.765544098214876
300036010,44.210526315789465,56.84210526315789,GX13+1,WINDOW2,27916.91218644381,157.4860776377307
300039010,14.736842105263158,31.57894736842105,4U1916-053,WINDOW2,93798.9349091053,21.45985987954541
300040010,18.94736842105263,35.78947368421052,4U1624-490,WINDOW2,60017.7530580461,30.989547346119032
300040020,18.94736842105263,35.78947368421052,4U1624-490,WINDOW2,52483.4983163476,30.417198761743972
300041010,6.315789473684211,18.94736842105263,SS433,WINDOW1,184122.0721244514,16.493009039934233
300049010,48.421052631578945,65.26315789473684,CYGNUS_X-1,WINDOW2BURST,15087.13245394826,383.5835615326299
300065010,44.210526315789465,56.84210526315789,Cyg_X-3,WINDOW2BURST,8114.140992671251,231.4848856701522
300072010,-1,-1,PDS456,WINDOW1,223864.1417592168,7.9161896410640225
300075010,-1,-1,NGC1365,WINDOW2,171307.5227504969,0.5482070985100834
900001010,27.36842105263158,40.0,4U1630-472,WINDOW2BURST,6015.331092238426,54.558925347178835
"""

# Load data into DataFrame
df = pd.read_csv(StringIO(data))

# Filter out rows where pileup fraction is -1
df = df[(df['pileup fraction 3 percent'] != -1) & (df['pileup fraction 1 percent'] != -1)]

# Set different marker styles for each mode
marker_styles = {'WINDOW2': 'o', 'WINDOW2BURST': 's', 'WINDOW1': 'D'}

# Plotting with different marker styles for each mode
plt.figure(figsize=(14, 8))

for mode, marker in marker_styles.items():
    subset = df[df['mode'] == mode]
    plt.scatter(subset['rate'], subset['pileup fraction 3 percent'], label=f'pileup fraction 3 percent - {mode}', marker=marker)
    plt.scatter(subset['rate'], subset['pileup fraction 1 percent'], label=f'pileup fraction 1 percent - {mode}', marker=marker)

    # Annotate points with OBSID and name, smaller font and diagonal text
    i = 0
    for _, row in subset.iterrows():
        plt.annotate(f"{row['OBSID']}\n{row['name']}", 
                     (row['rate'], row['pileup fraction 3 percent']), 
                     textcoords="offset points", 
                     xytext=(5,5), 
                     ha='right', 
                     fontsize=8,
                     rotation=40)
        plt.annotate(f"{row['OBSID']}\n{row['name']}", 
                     (row['rate'], row['pileup fraction 1 percent']), 
                     textcoords="offset points", 
                     xytext=(5,5), 
                     ha='right', 
                     fontsize=8,
                     rotation=40)

# Labels and title
plt.xlabel('Total Count Rate (c/s)')
plt.ylabel('Radius (pixel) at Pileup Fraction 3% and 1%')
plt.xscale("log")
plt.legend()
plt.title('Xtend Pileup Fractions vs. Count Rate')
plt.grid(True)
plt.savefig("xtend_pileup_plotresult_20040522.png")
plt.show()
