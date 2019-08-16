# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 19:26:40 2019

@author: dilip
"""

#from pyspark import SparkContext

import findspark
findspark.init()
import pyspark # only run after findspark.init()

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

conf = pyspark.SparkConf().setAppName('TriangleStrangle').setMaster('local[*]')
sc = pyspark.SparkContext(conf=conf)
spark = SparkSession(sc)

from pyspark.sql.functions import pandas_udf, PandasUDFType # needed to use pandas UDF functions
from pyspark.sql.types import MapType, StructType, ArrayType, StructField # might be useful in converitng dtypes
#spark.conf.set("spark.sql.execution.arrow.enabled", "true") # setting this here to impove the performance of toPandas()
spark.conf.set("spark.sql.execution.arrow.enabled", "false") 

import numpy as np
from scipy.spatial import Delaunay
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from IPython.display import display

import plotly as py
#import chart_studio.plotly as psi
import plotly.graph_objects as go
import matplotlib.cm as cm
from plotly.offline import iplot
import pyarrow


df4 = spark.read.csv("C:\\DJ\\Jacobs\\4 Sem\\Thesis\\sample data\\for_test\\house.csv", header = True, inferSchema = True)
df4.show(9)
df4.rdd.getNumPartitions()

df_pandas = df4.select("*").toPandas()
xpanda = df_pandas['X'].to_numpy()
ypanda = df_pandas['Y'].to_numpy()
zpanda = df_pandas['Z'].to_numpy()
xypandas = df_pandas[['X', 'Y']].to_numpy()  # for providing in delaunay


triang = mtri.Triangulation(xpanda, ypanda, triangles=None)
fig, ax = plt.subplots(subplot_kw =dict(projection="3d"))
ax.plot_trisurf(triang, zpanda)
plt.title('house Delaunay triangulation')
plt.show()



#-------------------------------------------------------------------------------------
#trying to pass mtri.triangualrtion to plot with plotly

x = xpanda.flatten()
y = ypanda.flatten()
z = zpanda
tri, args, kwargs = mtri.Triangulation.get_from_args_and_kwargs(x, y, z)
triangles = tri.get_masked_triangles()
xt = tri.x[triangles][..., np.newaxis]
yt = tri.y[triangles][..., np.newaxis]
zt = z[triangles][..., np.newaxis]

verts = np.concatenate((xt, yt, zt), axis=2)

print(verts)

#-----------------------------------------------------------------------------
fig = go.Figure(data=[go.Mesh3d(x=x, y=y, z=z,
                   alphahull=-1,
                   opacity=0.4,
                   color='cyan')])
fig.show()




#-----------------------------------------------------------------------------
sp_triag = Delaunay(xypandas)

# plotly set up
# defining the coloring
def map_z2color(zval, colormap, vmin, vmax):
    #map the normalized value zval to a corresponding color in the colormap

    if vmin>vmax:
        raise ValueError('incorrect relation between vmin and vmax')
    t=(zval-vmin)/float((vmax-vmin))#normalize val
    R, G, B, alpha=colormap(t)
    return 'rgb('+'{:d}'.format(int(R*255+0.5))+','+'{:d}'.format(int(G*255+0.5))+\
           ','+'{:d}'.format(int(B*255+0.5))+')'
    
#defining the indicies
def tri_indices(simplices):
    #simplices is a numpy array defining the simplices of the triangularization
    #returns the lists of indices i, j, k

    return ([triplet[c] for triplet in simplices] for c in range(3))

# define trisurf
def plotly_trisurf(x, y, z, simplices, colormap=cm.RdBu, plot_edges=None):
    #x, y, z are lists of coordinates of the triangle vertices 
    #simplices are the simplices that define the triangularization;
    #simplices  is a numpy array of shape (no_triangles, 3)
    #insert here the  type check for input data

    points3D=np.vstack((x,y,z)).T
    tri_vertices=map(lambda index: points3D[index], simplices)# vertices of the surface triangles     
    zmean=[np.mean(tri[:,2]) for tri in tri_vertices ]# mean values of z-coordinates of 
                                                      #triangle vertices
    min_zmean=np.min(zmean)
    max_zmean=np.max(zmean)
    facecolor=[map_z2color(zz,  colormap, min_zmean, max_zmean) for zz in zmean]
    I,J,K=tri_indices(simplices)

    triangles=go.Mesh3d(x=x,
                     y=y,
                     z=z,
                     facecolor=facecolor,
                     i=I,
                     j=J,
                     k=K,
                     name=''
                    )

    if plot_edges is None:# the triangle sides are not plotted 
        return [triangles]
    else:
        #define the lists Xe, Ye, Ze, of x, y, resp z coordinates of edge end points for each triangle
        #None separates data corresponding to two consecutive triangles
        lists_coord=[[[T[k%3][c] for k in range(4)]+[ None]   for T in tri_vertices]  for c in range(3)]
        Xe, Ye, Ze=[reduce(lambda x,y: x+y, lists_coord[k]) for k in range(3)]

        #define the lines to be plotted
        lines=go.Scatter3d(x=Xe,
                        y=Ye,
                        z=Ze,
                        mode='lines',
                        line=dict(color= 'rgb(0,0,0)', width=1.5)
                        #line=dict(color= 'rgb(50,50,50)', width=1.5)
               )
        return [triangles, lines]
    

# setting the layout using the standard will update later based on requirement
axis = dict(
showbackground=True,
backgroundcolor="rgb(230, 230,230)",
gridcolor="rgb(255, 255, 255)",
zerolinecolor="rgb(255, 255, 255)",
    )

layout = go.Layout(
         title='default title',
         width=800,
         height=800,
         scene=dict(
         xaxis=dict(axis),
         yaxis=dict(axis),
         zaxis=dict(axis),
        aspectratio=dict( x=1, y=1, z=0.5),
        )
        )
         
         
# plot figure by updting the parameters
#'''       
data2 = plotly_trisurf(xpanda, ypanda, zpanda, sp_triag.simplices, colormap=cm.cubehelix, plot_edges=None)      

# updating the layout for plotting
fig2 = go.Figure(data=data2, layout = layout)
fig2['layout'].update(dict(title='Triangulated surface',
                          scene=dict(camera=dict(eye=dict(x=1.75,
                                                          y=-0.7,
                                                          z= 0.75)
                                                )
                                    )))

py.offline.iplot(fig2, filename='trisurf-surface-pyspark')

#'''

'''
import os

if not os.path.exists("images"):
    os.mkdir("images")
    
    
#fig.write_image("images/TIN.png") 

'''





sc.stop()