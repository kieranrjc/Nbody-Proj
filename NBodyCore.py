import numpy as np, matplotlib.pyplot as plt, matplotlib.animation as anim
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import cnames
from numba import jit

@jit                                            #using numba to speed up solving of large systems.
def NbodyGeneral(A,T,B):                        #DOES SLOW DOWN CODE FOR LOW VALUES OF N DUE TO
                                                #CACHING AND MACHINE CODE SET UP DISABLE IF NECESSARY
    N   = B[0]
    G   = B[1]
    M   = np.zeros(shape=(N))

    x0  = np.zeros(shape=(N,)); y0  = np.zeros(shape=(N,)); z0  = np.zeros(shape=(N,))      #Simultaneous assignment causes bugs in return list. 
    vx0 = np.zeros(shape=(N,)); vy0 = np.zeros(shape=(N,)); vz0 = np.zeros(shape=(N,))      #Pre-allocation of space.      
        
    fvx = np.zeros(N); fvy = np.zeros(N); fvz = np.zeros(N)    
    fx  = np.zeros(N); fy  = np.zeros(N); fz  = np.zeros(N)
    
    rx  = np.zeros(shape=(N,N)); ry  = np.zeros(shape=(N,N)); rz  = np.zeros(shape=(N,N))    
    r   = np.zeros(shape=(N,N))
    
    retlist = np.array([])
    
    for i in range(N):                      #Looping through N, pulling each initial condition and
        x0[i] = A[i*6]                      #separating it into the appropriate array.
        y0[i] = A[(i*6)+1]
        z0[i] = A[(i*6)+2]
        vx0[i]= A[(i*6)+3]
        vy0[i]= A[(i*6)+4]
        vz0[i]= A[(i*6)+5]
        M[i]  = B[2+i]
        
    for i in range(N):                      #calculation of the separation between each body in
        rx[i,:] = x0[:]-x0[i]               #each dimension
        ry[i,:] = y0[:]-y0[i]    
        rz[i,:] = z0[:]-z0[i]
        
    r[:] = np.sqrt(rx[:]**2+ry[:]**2+rz[:]**2)  #calculation of the total direct distance between 
                                                #each body
    fvx[:] = vx0[:]
    fvy[:] = vy0[:]                             #initial velocity can be returned straight
    fvz[:] = vz0[:]

    for i in range(N):
        for j  in range(N):                     #summation loop to calculate acceleration contribution
            if i != j:                          #per body; i=j avoided to prevent singularity in r^3
                
                fx[i]  += G*M[j]*rx[i,j]/(r[i,j]**3)
                fy[i]  += G*M[j]*ry[i,j]/(r[i,j]**3)
                fz[i]  += G*M[j]*rz[i,j]/(r[i,j]**3)
        
    for i in range(N):
        retlist = np.append(retlist,[fvx[i],fvy[i],fvz[i],fx[i],fy[i],fz[i]])   #creation of return list in order specified by 
                                                                                #the order of initial conditions
    return retlist
    
def EnergyCheck(solp,B,T):
    
    N = B[0]                                        #moving additional arguments to arrays
    G = B[1]
    M = B[2:]

    KE    = np.zeros(shape = (N,solp.shape[1]))     #pre-allocation of space
    GP    = np.zeros(shape = (N,solp.shape[1]))     #again, simultaneous assignment caused bugs 
    x     = np.zeros(shape = (N,solp.shape[1]))
    y     = np.zeros(shape = (N,solp.shape[1]))
    z     = np.zeros(shape = (N,solp.shape[1]))
    vx    = np.zeros(shape = (N,solp.shape[1]))
    vy    = np.zeros(shape = (N,solp.shape[1]))
    vz    = np.zeros(shape = (N,solp.shape[1]))
    KEtot = np.zeros(shape = (1,solp.shape[1]))
    GPtot = np.zeros(shape = (1,solp.shape[1]))
    rx    = np.zeros(shape = (N,N,solp.shape[1]))
    ry    = np.zeros(shape = (N,N,solp.shape[1]))
    rz    = np.zeros(shape = (N,N,solp.shape[1]))
    r     = np.zeros(shape = (N,N,solp.shape[1]))
    Etot  = np.zeros(shape = (N,solp.shape[1]))
    
    for i in range(N):                              #creating N x tn array to store positions and velocities
        x[i] = solp[(i*6)]                          #at each time step
        y[i] = solp[(i*6)+1]
        z[i] = solp[(i*6)+2]
        vx[i]= solp[(i*6)+3]
        vy[i]= solp[(i*6)+4]
        vz[i]= solp[(i*6)+5]
        
    for i in range(N):
        rx[:,i,:] = x[:] - x[i]                     #similar to core nbody solver, except a third dimension
        ry[:,i,:] = y[:] - y[i]                     #is required for time
        rz[:,i,:] = z[:] - z[i]

    for i in range(N):
        r[:,i,:] = np.sqrt(rx[:,i,:]**2+ry[:,i,:]**2+rz[:,i,:]**2)      #same again, but with time.
        
    for i in range(N):                                                  #calculation of kinetic energy
        KE[i] = 0.5*M[i]*(vx[i]**2+vy[i]**2+vz[i]**2)                   #for each body
    
    for i in range(N):
        for j in range(N):
            if i!=j:                                                    #Calculation of Gravitation
                GP[i] += -G*M[i]*M[j]/r[i,j,:]                          #potential per body
    
    for i in range(N):                                                  #Summing total KE and GPE
        GPtot[:] += GP[i]                                               #for the system
        KEtot[:] += KE[i]
    
    Etot = GPtot + 2*KEtot                                                #Total energy of the system, using virial theorem.
    
    fig1 = plt.figure()
    ax = fig1.add_subplot(111)
    
    ke,   = ax.plot(T,KEtot.T,'r-',label='Total Kinetic Energy')                   #plotting of KE, GPE, and total energy 
    gp,   = ax.plot(T,GPtot.T,'b-',label='Total Gravitational Potential Energy')   #against time
    etot, = ax.plot(T,Etot.T,'g-',label='Total energy of the system')
    bind, = ax.plot(T,[0]*len(T),'k:',label='"Binding" point')                      #extra line for zero point for binding
    
    ax.legend(handles=[ke,gp,etot,bind],prop={'size':10})                           #legend formatting
    ax.set_ylabel('Energy - J')
    ax.set_xlabel('Time - s')
    ax.set_title('Energy against time for the system')   
    
    return 
    
     
    
    
def Trajectories(solp,tn,intv,lim,N,centreb):
    
    xmin,xmax,ymin,ymax,zmin,zmax = lim
    
    fig1 = plt.figure(1)                                                            #setting up figure and 3d axes 
    ax1 = fig1.add_axes([0,0,0.5,1],aspect='equal',projection='3d'  )    

    ax2 = fig1.add_axes([0.5, 0, 0.5, 1],aspect='equal', projection='3d')
    ax2.axis('off')
    
    colors = plt.cm.jet(np.linspace(0,1,N))                                         #4xN array of RGBA values, allowing
                                                                                    #for a unique colour per body
    
    lines = sum([ax2.plot([], [], [], '-', c=c,linewidth=0.75) for c in colors], []) #creating collection of lines/points 
    pts   = sum([ax2.plot([], [], [], 'o', c=c) for c in colors], [])               #using unique colours. Thinner lines for clearer plots.
    
    ax1.view_init(60,45)                                                            #setting the same initial vewing angle on each axis
    ax2.view_init(60,45)
            
    ax1.set_xlim((xmin,xmax))
    ax1.set_ylim((ymin,ymax))
    ax1.set_zlim((zmin,zmax))
    ax2.set_xlim((xmin,xmax))                                                       #setting axis limits to pre-defined variables.
    ax2.set_ylim((ymin,ymax))
    ax2.set_zlim((zmin,zmax))

    solfix = np.vstack(([solp[i*6:i*6+3] for i in range(N)]))                       #reformatting of solution array,
    solstack = np.swapaxes(np.dstack(([solfix[i*3:i*3+3] for i in range(N)])),0,2)  #removing velocity, and 3d stacking for time.
    
    def init():                                                                     #animation initialisation, blanking line/point data
        for line, pt in zip(lines, pts):
            line.set_data([], [])
            line.set_3d_properties([])
    
            pt.set_data([], [])
            pt.set_3d_properties([])
        return lines + pts
        
    def animate(i):                                                         #animation function
        
        i = ((tn//1800)*i)%tn                                               #frame skip utility, to speed up final animations.
                                                                            #The skip is set to a ratio of tn/1800 to allow for 
                                                                            #--TN MUST BE AT LEAST 1800 FOR THIS TO WORK--
                                                                            #complete animation in 30 seconds at 60fps. 
        if centreb != None:                                                 
            ax2.set_xlim((xmin+solp[centreb,i],xmax+solp[centreb+0,i]))
            ax2.set_ylim((ymin+solp[centreb+1,i],ymax+solp[centreb+1,i]))   #functionality for having a body as animation centre
            ax2.set_zlim((zmin+solp[centreb+2,i],zmax+solp[centreb+2,i]))     
            
        for line, pt, soli in zip(lines, pts, solstack):                    #setting and returning the frame data from collated zip array
            x,y,z=soli[:i].T
            line.set_data(x,y)
            line.set_3d_properties(z)            
    
            pt.set_data(x[-1:],y[-1:])
            pt.set_3d_properties(z[-1:])        
        
        return lines + pts
        
    ani = anim.FuncAnimation(fig1,animate,init_func=init,frames=1800,interval=intv)   #inbuilt function to wrap up and display frames.
    
    for i in range(N):
        ax1.plot(solp[i*6],solp[i*6+1],solp[i*6+2],c=colors[i],linewidth=0.75)       #plotting entire     
    
    return ani