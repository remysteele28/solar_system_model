%pip install -q matplotlib

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import json                                                                                                                 # converts java object to python object
import requests                                                                                                             # requests data from API
from matplotlib.patches import Ellipse, Circle, PathPatch
from ipywidgets import interact, interactive, fixed, interact_manual, widgets
import ipywidgets as widgets
import math

plt.rcParams['axes.facecolor'] = 'black'                                                                                    # Black background on plot

def c(x):
    return [np.random.default_rng(x**2+1).random(), np.random.default_rng(x**2+2).random(), np.random.default_rng(x**2+13).random()]

def stars(plt,Zoom):
    N = 300000
    n = N*(1-.997*(Zoom-25)/4400)
    x = (np.random.default_rng(5).random(int(n))-1/2)*9e9
    y = (np.random.default_rng(6).random(int(n))-1/2)*9e9
    plt.scatter(x,y,color='w',s=.2,alpha=.4*np.random.default_rng(7).random(int(n)))

def stars3D(ax,Zoom):
    N = 300000
    n = N*(1-.997*(Zoom-50)/4100)
    x = (np.random.default_rng(5).random(int(n))-1/2)*1e10
    y = (np.random.default_rng(6).random(int(n))-1/2)*1e10
    z = (np.random.default_rng(7).random(int(n))-1/2)*1e10
    ax.scatter3D(x,y,z,'.',s=.2,color='w',alpha=.4*np.random.default_rng(7).random(int(n)))

def degtorad(deg):
    return deg*np.pi/180

def semiminor(a,e):                                                                                                         # Calculate semiminor axis from semimajor and eccentricity
    return a*np.sqrt(1-e**2)

class Vector:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y, self.z + other.z)

    def __mul__(self, other):
        return Vector(self.x * other, self.y * other, self.z * other)

    def __div__(self, other):
        return Vector(self.x / other, self.y / other, self.z / other)

    def __str__(self):
        return '({x}, {y}, {z})'.format(x=self.x, y=self.y, z=self.z)

    def abs(self):
        return math.sqrt(self.x*self.x + self.y*self.y + self.z*self.z)

def mulAdd(v1, x1, v2, x2):
    return v1 * x1 + v2 * x2

def rotate(i, j, alpha):
    return [mulAdd(i,math.cos(alpha),j,math.sin(alpha)), mulAdd(i,-math.sin(alpha),j,math.cos(alpha))]

def orbitalStateVectors(semimajorAxis, eccentricity, inclination, longitudeOfAscendingNode, argumentOfPeriapsis, trueAnomaly):
    i = Vector(1, 0, 0)
    j = Vector(0, 1, 0)
    k = Vector(0, 0, 1)

    p = rotate(i, j, longitudeOfAscendingNode)
    i = p[0]
    j = p[1]
    p = rotate(j, k, inclination)
    j = p[0]
    p  =rotate(i, j, argumentOfPeriapsis)
    i = p[0]
    j = p[1]

    l = 2.0 if (eccentricity == 1.0) else 1.0 - eccentricity * eccentricity
    l *= semimajorAxis
    c = math.cos(trueAnomaly)
    s = math.sin(trueAnomaly)
    r = l / (1.0 + eccentricity * c)
    rprime = s * r * r / l
    position = mulAdd(i, c, j, s) * r
    speed = mulAdd(i, rprime * c - r * s, j, rprime * s + r * c)

    return [position, speed]

def plot_ellipse(semimaj=1,semimin=1,phi=0,x_cent=0,y_cent=0,theta_num=int(1e3),ax=None,plot_kwargs=None,\
                    fill=False,fill_kwargs=None,cov=None,mass_level=0.68,name='0',col='y'):

    # Generate data for ellipse structure
    theta = np.linspace(0,2*np.pi,theta_num)                                    # Copied this from somewhere
    r = 1 / np.sqrt((np.cos(theta))**2 + (np.sin(theta))**2)
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    data = np.array([x,y])
    S = np.array([[semimaj,0],[0,semimin]])
    R = np.array([[np.cos(phi),-np.sin(phi)],[np.sin(phi),np.cos(phi)]])
    T = np.dot(R,S)
    data = np.dot(T,data)
    data[0] += x_cent
    data[1] += y_cent
    
    if fill == True:
        plt.fill(data[0],data[1],facecolor='y')                                 # Only used for sun, which is actual size

    plt.plot(data[0],data[1],color=col,linestyle='--',label=name,alpha=.67)    # Use custom random colors

def sphere(r,x,y,z,ax,name,color='y'):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    xs = r * np.outer(np.cos(u), np.sin(v))
    ys = r * np.outer(np.sin(u), np.sin(v))
    zs = r * np.outer(np.ones(np.size(u)), np.cos(v))

    # Plot the surface
    ax.plot_surface(xs+x, ys+y, zs+z, color=color)
    ax.scatter3D(0,0,0, color=color, label = name)

response = requests.get("https://api.le-systeme-solaire.net/rest/bodies/").json()                                           # Request data from https://api.le-systeme-solaire.net/en/ and
                                                                                                                            # Convert it to a python object
planets = {}                                                                                                                # Planet dictionary containg pertinant information
k = 0                                                                                                                       # k = 0, 1, 2...                                                                               
menu = ['Sun']                                                                                                              # What body do you want to look at?

for i in range(len(response["bodies"])):                                                                                    # Search for planets
    
    if response["bodies"][i]["isPlanet"] == True:                                                                           # Fill information about each planet

        names = response["bodies"][i]["englishName"]

        planets[k] = {\
                      

        'Name' : names,\
        'Semimajor Axis' : response["bodies"][i]["semimajorAxis"],\
        'Eccentricity' : response["bodies"][i]["eccentricity"],\
        'Semiminor Axis' : semiminor(response["bodies"][i]["semimajorAxis"],response["bodies"][i]["eccentricity"]),\
        'Inclination' : degtorad(response["bodies"][i]["inclination"]),\
        'Periapsis' : degtorad(response["bodies"][i]["argPeriapsis"]),\
        'LAN' : degtorad(response["bodies"][i]["longAscNode"]),\
        'Anomoly' : degtorad(response["bodies"][i]["mainAnomaly"]),\
        'Radius' : response["bodies"][i]["meanRadius"],\
        
        }
        menu.append(response["bodies"][i]["englishName"])                                                                   # Fill list for dropdown menu (includes sun)
        k = k + 1
        
    if response["bodies"][i]["englishName"] == 'Sun':                                                                       # Fill sun attributes
        sun_radius = response["bodies"][i]["meanRadius"]

x,y,z,r = np.empty(0),np.empty(0),np.empty(0),np.empty(0)
xt = {}

for i in range(len(planets)):
    a = planets[i]["Semimajor Axis"]
    e = planets[i]["Eccentricity"]
    b = semiminor(a,e)
    I = planets[i]["Inclination"]
    omega_AP = planets[i]["Periapsis"]
    omega_LAN = planets[i]["LAN"]
    T = planets[i]["Anomoly"]
    r = np.append(r, planets[i]["Radius"])

    x = np.append(x, orbitalStateVectors(a,e,I,omega_LAN,omega_AP,T)[0].x)
    y = np.append(y, orbitalStateVectors(a,e,I,omega_LAN,omega_AP,T)[0].y)
    z = np.append(z, orbitalStateVectors(a,e,I,omega_LAN,omega_AP,T)[0].z)

    around = np.linspace(0,2*np.pi,int(a/500000))
    x_t,y_t,z_t = np.empty(0),np.empty(0),np.empty(0)

    for j in range(len(around)):
        x_t = np.append(x_t, orbitalStateVectors(a,e,I,omega_LAN,omega_AP,around[j])[0].x)
        y_t = np.append(y_t, orbitalStateVectors(a,e,I,omega_LAN,omega_AP,around[j])[0].y)
        z_t = np.append(z_t, orbitalStateVectors(a,e,I,omega_LAN,omega_AP,around[j])[0].z)
    
    xt[i] = {0 : x_t, 1 : y_t, 2 : z_t}


def solar_system3d(Zoom,Incline,Rotation,Center):                               # Plot function
    semiargx = 0                                                                # 'Camera' position initialization
    semiargy = 0                                                                #    "         "          "
    semiargz = 0                                                                #    "         "          "

    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(projection='3d')
    stars3D(ax,Zoom)
    
    for i in range(len(planets)):
        
        ax.plot3D(xt[i][0],xt[i][1],xt[i][2], '--', alpha = .4, color=c(i))
        sphere(1e4*(r[i])**(1/2)*Zoom**(1/2),x[i],y[i],z[i],ax,color=c(i), name = planets[i]["Name"])
        
    if Center == planets[i]["Name"]:                                                                                                                                            # Center camera to object of choice
        semiargx = x                                                                                                  
        semiargy = y
        semiargz = z

    sphere(35*sun_radius,0,0,0,ax,'Sun')
    ax.legend(labelcolor='linecolor',loc='upper right')
    ax.set_xlabel('x (kpc)')
    ax.set_ylabel('y (kpc)')
    ax.set_zlabel('z (kpc)')
    ax.set_title('Solar System', c='k')
    ax.view_init(Incline, Rotation)
    ax.set_xlim(-Zoom*sun_radius+semiargx,Zoom*sun_radius+semiargx)
    ax.set_ylim(-Zoom*sun_radius+semiargy,Zoom*sun_radius+semiargy)
    ax.set_zlim(-Zoom*sun_radius+semiargz,Zoom*sun_radius+semiargz)
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)) 
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)) 
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0)) 
    ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0)) 
    ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.set_box_aspect([1,1,1])
    ax.set_xticks([])                               
    ax.set_yticks([])                               
    ax.set_zticks([])
    plt.show()
    

interact(solar_system3d,\
         
         Zoom = widgets.FloatSlider(value=400,min=50,max=4100,step=1),\
         Incline = widgets.FloatSlider(value=90,min=0,max=90,step=1),\
         Rotation = widgets.FloatSlider(value=0,min=0,max=360,step=1),\
         Center=menu)