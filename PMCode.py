import numpy as np, matplotlib.pyplot as plt

plt.close('all')

n = 2000                            #number of bodies

Tmax = 100000000000000              #tmax has to be very large for the 'smoothed' attempts,
h    = 10000000                     #due to h also being very large. 
t = 0

ng = 512
x=np.linspace(0,ng,ng+1)            #x and y were created in order to calculate k
y=np.linspace(0,ng,ng+1)            #for the 1/k^2 green's function. kx and ky
kx=2*np.pi*x[:]                     #were made into a grid, and then squared and 
ky=2*np.pi*y[:]                     #summed
kxg,kyg = np.meshgrid(kx,ky)
kg = np.sqrt(kxg[:]**15+kyg[:]**15)
kg[0,0] = 1                         #Temporary assignment to avoid a singularity.

pg = np.zeros(shape=(ng+1,ng+1))

p = np.zeros(shape=(2*n,2))           #uniform ring created by calculating points at a 
                                    #constant radius using sin&cos, with a multiple of 2*pi,
for i in range(n//2):                  #as the ratio between point number, and total number of points.
    p[i,0] = np.cos((i/n)*2*np.pi)*1/6*ng
    p[i,1] = np.sin((i/n)*2*np.pi)*1/6*ng

    p[i+n,0] = np.cos((i/n)*2*np.pi)*1/9*ng
    p[i+n,1] = np.sin((i/n)*2*np.pi)*1/9*ng
    
p += 0.5*ng

vxhalf = np.zeros(shape=(len(p),))
vyhalf = np.zeros(shape=(len(p),))

pngp = (np.around(p))               #rounding position of points to the nearets grid point

pngpint = pngp.astype(int)          #A set of rounded points in integer form, to allow for iteration

m = 1
G = 1             

for i,j in pngpint:                 #mapping the density by taking the rounded x,y co-ordinates of every
    pg[i,j]+=m                      #particle and adding it's mass to the grid point. 

        
fig1 = plt.figure(1)
ax1 = fig1.add_axes([0,0,0.5,1],aspect='equal',autoscale_on=False,xlim=(1/8*ng,7/8*ng),ylim=(1/8*ng,7/8*ng))  #creating the figure window, and side by side axes.
ax2 = fig1.add_axes([0.5,0,0.5,1],aspect='equal',autoscale_on=False,xlim=(1/8*ng,7/8*ng),ylim=(1/8*ng,7/8*ng))

ax1.axis('off')
ax2.axis('off')

map1 = ax1.imshow(pg)               #the density heat map

points, = ax2.plot(p[:,0],p[:,1],'.',markersize=0.5) #the explicit particle position plot
plt.show(block=False)

fft = np.fft.fft2(pg)               #Fourier transform of the density grid.

fft2 = -4*np.pi*G*fft[:]/kg[:]**2   #Applying constants and green's function, to get the Potential

fft2[0,0] = 0                       #Manually assigning what would be a singularity to 0.

ifft = np.fft.ifft2(fft2)           #Inverse fourier transform of the Potential.

Fxinit,Fyinit = np.gradient(-ifft)  #Taking the gradient along both axes, and separating to x,y

vxhalf = vxhalf[:] + 0.5*h* Fxinit[pngpint[:,0],pngpint[:,1]]/m #Velocity Verlet to get half step
vyhalf = vyhalf[:] + 0.5*h* Fyinit[pngpint[:,0],pngpint[:,1]]/m

while t<=Tmax:            
    
    pxstep = p[:,0] + h*vxhalf[:]   #Updated position for x and y
    pystep = p[:,1] + h*vyhalf[:]
    
    p = np.vstack((pxstep,pystep)).T #Recompiling the positions for density mapping
    
    pg = np.zeros(shape=(ng+1,ng+1))
    
    pngp = np.around(p)
    pngpint = pngp.astype(int)
    
    pngploop = pngpint              #Separate integer lists, one for deletion, one for looping. 
    shift = 0
    for i,j in pngploop:
        if ((0<=i<=ng) and (0<=j<=ng)):                     # The loop density mapping function
            pg[i,j]+=m                                      #requires boundary conditions for 
        else:                                               #particles leaving the grid. This loop
            pngpint = np.delete(pngpint,i-shift,axis=0)     #would check if x and y were in range
            vxhalf  = np.delete(vxhalf,i-shift,axis=0)      #before mapping, and delete it otherwise.
            vyhalf  = np.delete(vyhalf,i-shift,axis=0)      #However for whatever reason this did    
            pxstep  = np.delete(pxstep,i-shift,axis=0)      #not work, regardless of attempts to fix
            pystep  = np.delete(pystep,i-shift,axis=0)      #it. 
            p       = np.delete(p,i-shift,axis=0)           #
            shift+=1    
    
    pg = np.rot90(np.flipud(pg),3)  #orientation of the plotting of the density grid and
                                    #points plot differed, due to plotting. 
    
    fftstep = np.fft.fft2(pg)
    fft2step = -4*np.pi*G*fftstep[:]/kg[:]**2
    fft2step[0,0] = 0
    ifftstep = np.fft.ifft2(fft2step)
    Fxstep,Fystep = np.gradient(-ifftstep)
    
    vxhalf = vxhalf[:] + h*Fxstep[pngpint[:,0],pngpint[:,1]]/m #full step velocity
    vyhalf = vyhalf[:] + h*Fystep[pngpint[:,0],pngpint[:,1]]/m

    plt.pause(0.001)                #pause to allow figure to draw new values. 
    points.set_xdata(p[:,0])    
    points.set_ydata(p[:,1])
    map1.set_array(pg)
    
    fig1.canvas.draw()
    
    t+=h                            #time counter, for the while loop
     
    



