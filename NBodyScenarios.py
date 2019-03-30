import numpy as np, matplotlib.pyplot as plt, scipy.integrate as spy, NBodyCore as nbc, matplotlib.animation as anim

plt.close('all')

AU = 1.469e11

def TestCase():                                         #Test case of equal unit masses, G=1 
                                                        #becomes unbound for separation distance >1, (-0.5,0.5)
    plt.close('all')    
    
    x1  = 0.4; x2  = -0.4
    y1  = 0  ; y2  = 0
    z1  = 0  ; z2  = 0
    vx1 = 0  ; vx2 = 0
    vy1 = 1  ; vy2 = -1
    vz1 = 0  ; vz2 = 0
    
    A = [x1,y1,z1,vx1,vy1,vz1,
         x2,y2,z2,vx2,vy2,vz2]
        
    M1 = 1
    M2 = 1
    
    N = 2
    G = 1
    
    tmax = 100
    tn = 1800
    T = np.linspace(0,tmax,tn)
    
    B = (N,G,M1,M2)                                     #Error with numba jit restricting B to be in a tuple
                                                        #(cannot unbox a list inside a tuple, as B is placed in, in the solver; (B,) )
    sol = spy.odeint(nbc.NbodyGeneral,A,T,args=(B,))
    solp = sol.T
    
    lim = [-1,1,-1,1,-1,1]
    ani = nbc.Trajectories(solp,tn,30,lim,N,None)       #All core functions moved into NBodyCore.py folder, and imported. 
    
#    FFwriter = anim.FFMpegWriter(fps = 60,extra_args=['-vcodec','libx264'])    #Used to save MP4s of animations;
#    ani.save('basic_animation.mp4',writer=FFwriter)                            #This can take a VERY long time, and is here
                                                                                #only for posterity
    nbc.EnergyCheck(solp,B,T)
    
    return ani   
    

def SolarSystem():                                  #simple set up of the solar system, N=10, using values 
    global AU                                       #of distance, orbital speed, and mass from Hyperphysics
    plt.close('all')                                #Done as a 'speed test of higher body simuations, 
    x1 = 0*AU ; x2 = -0.387*AU ; x3 = 0.723*AU      #and to test integrity of animations
    x4 = 1*AU ; x5 = 1.666*AU ; x6 = -5.23*AU ;
    x7 = 9.582*AU ; x8 = -19.2*AU ; x9 = 30.05*AU ;
    x10 = 37.75*AU
    
    y1 = 0*AU ; y2 = 0*AU ; y3 = 0*AU
    y4 = 0*AU ; y5 = 0*AU ; y6 = 0*AU
    y7 = 0*AU ; y8 = 0*AU ; y9 = 0*AU
    y10 = 0*AU
    
    z1 = 0*AU ; z2 = 0*AU ; z3 = 0*AU
    z4 = 0*AU ; z5 = 0*AU ; z6 = 0*AU
    z7 = 0*AU ; z8 = 0*AU ; z9 = 0*AU
    z10 = 11.61*AU
    
    vx1 = 0 ; vx2 = 0 ; vx3 = 0
    vx4 = 0 ; vx5 = 0 ; vx6 = 0
    vx7 = 0 ; vx8 = 0 ; vx9 = 0
    vx10 = 0
    
    vy1 = 0 ; vy2 =  -47.4e3 ; vy3 = 35e3
    vy4 = 29.8e3 ; vy5 = 24.1e3 ; vy6 = -13.1e3
    vy7 = 9.6e3 ; vy8 = -6.8e3 ; vy9 = 5.4e3
    vy10 = 4.74e3
    
    vz1 = 0 ; vz2 = 0 ; vz3 = 0
    vz4 = 0 ; vz5 = 0 ; vz6 = 0
    vz7 = 0 ; vz8 = 0 ; vz9 = 0
    vz10 = 0
        
    A = [x1,y1,z1,vx1,vy1,vz1,              #Prime example of 'table formatting' for initial conditions
         x2,y2,z2,vx2,vy2,vz2,
         x3,y3,z3,vx3,vy3,vz3,
         x4,y4,z4,vx4,vy4,vz4,
         x5,y5,z5,vx5,vy5,vz5,
         x6,y6,z6,vx6,vy6,vz6,
         x7,y7,z7,vx7,vy7,vz7,
         x8,y8,z8,vx8,vy8,vz8, 
         x9,y9,z9,vx9,vy9,vz9,
         x10,y10,z10,vx10,vy10,vz10]  
         
    M1 = 1.989e30 ; M2 = 3.3e23  ; M3 = 4.87e24  
    M4 = 5.976e24 ; M5 = 6.42e23 ; M6 = 1.9e27 
    M7 = 5.69e26  ; M8 = 8.68e25 ; M9 = 1.03e26
    M10 = 1.46e22 
    
    N = 10
    G = 6.67408e-11              
    B = (N,G,M1,M2,M3,M4,M5,M6,M7,M8,M9,M10) 
    
    tmax = 250*np.pi*1e7                        #Enough time to see ~1 full orbit of pluto, 
    tn   = 50000                                #Pluto included to show inclination in 3D.
    T    = np.linspace(0,tmax,tn)               #Rough estimate of time of 1 year taken as 1e7*pi
        
    sol = spy.odeint(nbc.NbodyGeneral,A,T,args=(B,))
    solp = sol.T   
    
    lim = [-5e12,5e12,-5e12,5e12,-2e12,2e12]            #Due to the way plotting is handled and viewing angle is handled,
    ani = nbc.Trajectories(solp,tn,30,lim,N,None)       #limits can afford to be within the trajectories, and gives
                                                        #the best 'zoom' to see everything.
                                                        
#    FFwriter = anim.FFMpegWriter(fps = 60,extra_args=['-vcodec','libx264'])    
#    ani.save('.mp4',writer=FFwriter)     

    nbc.EnergyCheck(solp,B,T)                            
    return ani
    
def Kozai(theta):                                       #Attempt at witnessing the kozai mechanism, to try
    global AU                                           #to see the effect of the Earth orbiting >40 degrees 
    plt.close('all')                                    #to the ecliptic. Theta can be submitted on runs.
    theta = 41*np.pi/180
    
    x1 = 0*AU ; x2 = -0.387*AU ; x3 = 0.723*AU                  #Tested with planets up to Jupiter,
    x4 = np.cos(theta)*AU ; x5 = 1.666*AU ; x6 = -5.23*AU ;     #as jupiter represents the other 'great mass'
    
    y1 = 0*AU ; y2 = 0*AU ; y3 = 0*AU
    y4 = 0*AU ; y5 = 0*AU ; y6 = 0*AU
    
    z1 = 0*AU ; z2 = 0*AU ; z3 = 0*AU
    z4 = np.sin(theta)*AU ; z5 = 0*AU ; z6 = 0*AU
    
    vx1 = 0 ; vx2 = 0 ; vx3 = 0
    vx4 = 0 ; vx5 = 0 ; vx6 = 0
    
    vy1 = 0 ; vy2 =  -47.4e3 ; vy3 = 35e3
    vy4 = 29.8e3 ; vy5 = 24.1e3 ; vy6 = -13.1e3
    
    vz1 = 0 ; vz2 = 0 ; vz3 = 0
    vz4 = 0 ; vz5 = 0 ; vz6 = 0
    
    A = [x1,y1,z1,vx1,vy1,vz1,  
         x2,y2,z2,vx2,vy2,vz2,
         x3,y3,z3,vx3,vy3,vz3,
         x4,y4,z4,vx4,vy4,vz4,
         x5,y5,z5,vx5,vy5,vz5,
         x6,y6,z6,vx6,vy6,vz6]
         
    M1 = 1.989e30 ; M2 = 3.3e23  ; M3 = 4.87e24  
    M4 = 5.976e24 ; M5 = 6.42e23 ; M6 = 1.9e27 
    
    N = 6
    G = 6.67408e-11              
    B = (N,G,M1,M2,M3,M4,M5,M6)
    
    tmax = 24*np.pi*1e7
    tn   = 50000
    T    = np.linspace(0,tmax,tn)
        
    sol = spy.odeint(nbc.NbodyGeneral,A,T,args=(B,))
    solp = sol.T
    
    lim = [-6*AU,6*AU,-6*AU,6*AU,-6*AU,6*AU]
    ani = nbc.Trajectories(solp,tn,30,lim,N,None)
    nbc.EnergyCheck(solp,B,T)
    
    return ani
    
def Passby(VStar):                              #Simulation of a 1 solar mass star passing by three stationary
                                                #bodies based on the three inner planets of the solar system.
    global AU                                   #Done as a test to see how this would fair as a theory for 
    plt.close('all' )                           #the creation of planetary systems. 
    x1 = -5*AU ; x2 = 0*AU ; x3 = 0*AU ;
    x4 = 0*AU ;                                 #Initival velocity of the star can be submitted, in km/s in console. 

    y1 = 0 ; y2 = 1*AU ; y3 =-0.6*AU ;
    y4 = -1.3*AU;

    z1 = 0 ; z2 =0.4*AU ; z3 =-1.1*AU ;
    z4 = 0.8*AU;

    vx1 = VStar*1e3 ; vx2 =0  ; vx3 =0 ;
    vx4 = 0 ;

    vy1 = 0; vy2 =0 ; vy3 =0 ;
    vy4 = 0;

    vz1 = 0; vz2 =0 ; vz3 =0 ;
    vz4 = 0;   
    
    A = [x1,y1,z1,vx1,vy1,vz1,  
         x2,y2,z2,vx2,vy2,vz2,
         x3,y3,z3,vx3,vy3,vz3,
         x4,y4,z4,vx4,vy4,vz4]
    
    M1 = 1.989e30 ; M2 = 3.3e23  ; M3 = 4.87e24  
    M4 = 5.976e24 ;
    N = 4
    G = 6.67408e-11              
    B = (N,G,M1,M2,M3,M4)
    
    tmax = 20*np.pi*1e7
    tn   = 20000
    T    = np.linspace(0,tmax,tn)    
    
    sol = spy.odeint(nbc.NbodyGeneral,A,T,args=(B,))
    solp = sol.T
    
    lim = [-5*AU,5*AU,-5*AU,5*AU,-2.5*AU,2.5*AU]
    ani = nbc.Trajectories(solp,tn,10,lim,N,0)
    nbc.EnergyCheck(solp,B,T)
    
    return ani
    
def Burrau(h):                              #Burrau's three body problem; time step to be submitted
    plt.close('all')                        #in console.
    
    x1  = -2; x2  = 2; x3  = 2
    y1  = -2; y2  = 1; y3  = -2 
    z1  = 0 ; z2  = 0; z3  = 0
    vx1 = 0 ; vx2 = 0; vx3 = 0
    vy1 = 0 ; vy2 = 0; vy3 = 0
    vz1 = 0 ; vz2 = 0; vz3 = 0 
       
    A = [x1,y1,z1,vx1,vy1,vz1,
         x2,y2,z2,vx2,vy2,vz2,
         x3,y3,z3,vx3,vy3,vz3]
    
    M1 = 3
    M2 = 4
    M3 = 5
        
    G = 1
    N = 3
    B = (N,G,M1,M2,M3)
    
    tmax = 60
    tn = int(tmax/h)                                    #Over a limited of period of motion the time step
    T = np.linspace(0,tmax,tn)                          #calculates the integer number of time steps.
    
    sol = spy.odeint(nbc.NbodyGeneral,A,T,args=(B,))
    solp = sol.T
    
    lim = [-7,7,-7,7,-2,2]
    ani = nbc.Trajectories(solp,tn,10,lim,N,None)
    nbc.EnergyCheck(solp,B,T)              
    return ani
    
def SolarSystemStarPass(VStar):                         #Low mass star, ~0.5 Solar, passing parallel to
    plt.close('all')                                    #the ecliptic, displaced 5AU in z.
    global AU
    
    x1 = 0*AU ; x2 = -0.387*AU ; x3 = 0.723*AU
    x4 = 1*AU ; x5 = 1.666*AU ; x6 = -5.23*AU ;
    x7 = 9.582*AU ; x8 = -19.2*AU ; x9 = 30.05*AU ;
    x10 = 37.75*AU ; x11 = 0*AU
    
    y1 = 0*AU ; y2 = 0*AU ; y3 = 0*AU
    y4 = 0*AU ; y5 = 0*AU ; y6 = 0*AU
    y7 = 0*AU ; y8 = 0*AU ; y9 = 0*AU
    y10 = 0*AU ; y11 = -50*AU
    
    z1 = 0*AU ; z2 = 0*AU ; z3 = 0*AU
    z4 = 0*AU ; z5 = 0*AU ; z6 = 0*AU
    z7 = 0*AU ; z8 = 0*AU ; z9 = 0*AU
    z10 = 11.61*AU ; z11 = 5*AU
    
    vx1 = 0 ; vx2 = 0 ; vx3 = 0
    vx4 = 0 ; vx5 = 0 ; vx6 = 0
    vx7 = 0 ; vx8 = 0 ; vx9 = 0
    vx10 = 0 ; vx11 = 0
    
    vy1 = 0 ; vy2 =  -47.4e3 ; vy3 = 35e3
    vy4 = 29.8e3 ; vy5 = 24.1e3 ; vy6 = -13.1e3
    vy7 = 9.6e3 ; vy8 = -6.8e3 ; vy9 = 5.4e3
    vy10 = 4.74e3 ; vy11 = VStar*1e3
    
    vz1 = 0 ; vz2 = 0 ; vz3 = 0
    vz4 = 0 ; vz5 = 0 ; vz6 = 0
    vz7 = 0 ; vz8 = 0 ; vz9 = 0
    vz10 = 0 ; vz11 = 0
        
    A = [x1,y1,z1,vx1,vy1,vz1,  
         x2,y2,z2,vx2,vy2,vz2,
         x3,y3,z3,vx3,vy3,vz3,
         x4,y4,z4,vx4,vy4,vz4,
         x5,y5,z5,vx5,vy5,vz5,
         x6,y6,z6,vx6,vy6,vz6,
         x7,y7,z7,vx7,vy7,vz7,
         x8,y8,z8,vx8,vy8,vz8, 
         x9,y9,z9,vx9,vy9,vz9,
         x10,y10,z10,vx10,vy10,vz10,
         x11,y11,z11,vx11,vy11,vz11]  
         
    M1 = 1.989e30 ; M2 = 3.3e23  ; M3 = 4.87e24  
    M4 = 5.976e24 ; M5 = 6.42e23 ; M6 = 1.9e27 
    M7 = 5.69e26  ; M8 = 8.68e25 ; M9 = 1.03e26
    M10 = 1.46e22 ; M11 = 1e30
    
    N = 11
    G = 6.67408e-11              
    B = (N,G,M1,M2,M3,M4,M5,M6,M7,M8,M9,M10,M11)
    
    tmax = 80*np.pi*1e7
    tn   = 10000
    T    = np.linspace(0,tmax,tn)
        
    sol = spy.odeint(nbc.NbodyGeneral,A,T,args=(B,))
    solp = sol.T   
    
    lim = [-15*AU,15*AU,-15*AU,15*AU,0,14*AU]

    ani = nbc.Trajectories(solp,tn,30,lim,N,0)
    
#    FFwriter = anim.FFMpegWriter(fps = 60,extra_args=['-vcodec','libx264'])
#    ani.save('.mp4',writer=FFwriter)
    
    nbc.EnergyCheck(solp,B,T)
    return ani
    
    
def ExoACBinary():                                          #Exoplanetary system around Alpha Centauri
    plt.close('all')                                        #Close profile: vy3 = 25e3, x3 = 10AU, tmax = <35
    global AU                                               #lims ~-1e12,1e12 x/y
                                                            #Wide profile: vy3 = 10e3, x3 = 25AU, tmax, 10,000-12,000
    x1 = -9*AU; x2 = 9*AU; x3 = 25*AU                       #lims ~-2e3,2e13 x/y
    
    y1 = 0; y2 = 0; y3 = 0
    
    z1 = 0; z2 = 0; z3 = 0
    
    vx1 = 0; vx2 = 0; vx3 = 0
    
    vy1 = 2e3; vy2 = -(0.9/1.1)*2e3; vy3 = 10e3 
    
    vz1 = 0; vz2 = 0; vz3 = 0
    
    A = [x1,y1,z1,vx1,vy1,vz1,
         x2,y2,z2,vx2,vy2,vz2,
         x3,y3,z3,vx3,vy3,vz3]
    
    M1 = 0.9*1.989e30; M2 = 1.1*1.989e30; M3 = 6e24
    G = 6.67408e-11
    N = 3
    B = (N,G,M1,M2,M3)     
    
    tmax = 10000*np.pi*1e7
    tn = 200000
    T = np.linspace(0,tmax,tn)   
    
    sol = spy.odeint(nbc.NbodyGeneral,A,T,args=(B,))
    solp = sol.T
    
    lim = [-2e13,2e13,-2e13,2e13,-1,1]
    ani = nbc.Trajectories(solp,tn,10,lim,N,None)
    
#    FFwriter = anim.FFMpegWriter(fps = 60,extra_args=['-vcodec','libx264'])
#    ani.save('ExoACWideOrbit.mp4',writer=FFwriter)
    
    nbc.EnergyCheck(solp,B,T)
    return ani
    
