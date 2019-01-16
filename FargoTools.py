import copy
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate


class Mesh():
    """
    Mesh class, for keeping all the mesh data.
    Input: directory [string] -> this is where the domain files are.
    """
    def __init__(self, directory=""):
        if len(directory) > 1:
            if directory[-1] != '/':
                directory += '/'
        try:
            domain_x = np.loadtxt(directory+"domain_x.dat")
        except IOError:
            print("IOError with domain_x.dat")
        try:
            #We avoid ghost cells
            domain_y = np.loadtxt(directory+"domain_y.dat")[3:-3]
        except IOError:
            print("IOError with domain_y.dat")
        self.xm = domain_x #X-Edge
        self.ym = domain_y #Y-Edge

        self.xmed = 0.5*(domain_x[:-1] + domain_x[1:]) #X-Center
        self.ymed = 0.5*(domain_y[:-1] + domain_y[1:]) #Y-Center

        #(Surfaces computed from edge to edge)
        #We first build 2D arrays for x & y (theta,r)
        T,R = np.meshgrid(self.xm, self.ym)
        R2  = R*R
        self.surf = 0.5*(T[:-1,1:]-T[:-1,:-1])*(R2[1:,:-1]-R2[:-1,:-1])


class Parameters():
    """
    Class for reading the simulation parameters.
    input: string -> name of the parfile, normally variables.par
    """
    def __init__(self, directory=""):
        if len(directory) > 1:
            if directory[-1] != '/':
                directory += '/'
        try:
            params = open(directory+"variables.par",'r') #Opening the parfile
        except IOError:                  # Error checker.
            print(paramfile + " not found.")
            return
        lines = params.readlines()     # Reading the parfile
        params.close()                 # Closing the parfile
        par = {}                       # Allocating a dictionary
        for line in lines:             #Iterating over the parfile
            name, value = line.split() #Spliting the name and the value (first blank)
            try:
                float(value)           # First trying with float
            except ValueError:         # If it is not float
                try:
                    int(value)         #                   we try with integer
                except ValueError:     # If it is not integer, we know it is string
                    value = '"' + value + '"'
            par[name] = value          # Filling the dictory
        self._params = par             # A control atribute, actually not used, good for debbuging
        for name in par:               # Iterating over the dictionary
            exec("self."+name.lower()+"="+par[name]) #Making the atributes at runtime


class Field(Mesh, Parameters):
    """
    Field class, it stores the mesh, parameters and scalar data
    for a scalar field.
    Input: field [string] -> filename of the field
           staggered='c' [string] -> staggered direction of the field.
                                      Possible values: 'x', 'y', 'xy', 'yx'
           directory='' [string] -> where filename is
           dtype='float64' (numpy dtype) -> 'float64', 'float32',
                                             depends if FARGO_OPT+=-DFLOAT is activated
    """
    def __init__(self, field, staggered='c', directory='', dtype='float64'):
        if len(directory) > 1:
            if directory[-1] != '/':
                directory += '/'
        Mesh.__init__(self, directory)       #All the Mesh attributes inside Field
        Parameters.__init__(self, directory) #All the Parameters attributes inside Field

        #Now, the staggering:
        if staggered.count('x') > 0:
            self.x = self.xm[:-1] #Do not dump the last element
        else:
            self.x = self.xmed
        if staggered.count('y') > 0:
            self.y = self.ym[:-1]
        else:
            self.y = self.ymed

        self.data = self.__open_field(directory + field, dtype) #The scalar data is here.

    def __open_field(self, f, dtype):
        """
        Reading the data
        """
        field = np.fromfile(f, dtype=dtype)
        return field.reshape(self.ny, self.nx)

    def plot(self, log=False, cartesian=False, cmap='magma', **karg):
        """
        A layer for plt.imshow or pcolormesh function.
        if cartesian = True, pcolormesh is launched.
        """
        ax = plt.gca()
        if log:
            data = np.log10(self.data)
        else:
            data = self.data
        if cartesian:
            interpolated = 0.5*(self.data[:,0] + self.data[:,-1])
            self.data = np.c_[self.data, interpolated]
            self.x = np.append(self.x, self.x[-1] + (self.x[-1] - self.x[-2]))
            T,R = np.meshgrid(self.x, self.y)
            X = R * np.cos(T)
            Y = R * np.sin(T)
            plt.pcolormesh(X, Y, data, cmap=cmap, **karg)
        else:
            T,R = np.meshgrid(self.x, self.y)
            plt.pcolormesh(T, R, data, cmap=cmap, **karg)
            #ax.imshow(data, cmap=cmap, origin='lower', aspect='auto',
            #          extent=[self.x[0], self.x[-1], self.y[0], self.y[-1]],
            #          **karg)

    #The same but for contours
    def contour(self, log=False, cartesian=False, **karg):
        if log:
            data = np.log10(self.data)
        else:
            data = self.data
        ax = plt.gca()
        T,R = np.meshgrid(self.x, self.y)
        if cartesian:
            X = R * np.cos(T)
            Y = R * np.sin(T)
            plt.contour(X, Y, data, **karg)
        else:
            plt.contour(T, R, data, **karg)

def shift_field(Field, direction):
    """
    Half-cell shifting along the direction ('x','y', 'xy', 'yx').

    Note that after a call to this function, Field.xm/xmed does not represent
    the coodinates of the field anymore (it is not hard to improve).
    """
    F = copy.deepcopy(Field)
    if direction.count('x') > 0:
        F.data = 0.5*(Field.data[:,1:] + Field.data[:,:-1])
        F.x = 0.5*(Field.x[1:] + Field.x[:-1])
    if direction.count('y') > 0:
        F.data = 0.5*(F.data[1:,:] + F.data[:-1,:])
        F.y = 0.5*(F.y[1:] + F.y[:-1])

    F.nx = len(F.x)
    F.ny = len(F.y)

    return F

def cut_field(Field, direction, side):
    """
    Cutting a field:
    Input: field --> a Field class
           axis  --> 'x', 'y' or 'xy'
           side  --> 'p' (plus), 'm' (minnus), 'pm' (plus/minnus)
    """
    _cut_field = copy.deepcopy(Field)
    ny,nx = Field.ny, Field.nx
    mx = my = px = py = 0

    if direction.count('x') > 0:
        if side.count('m') > 0:
            mx = 1
        if side.count('p') > 0:
            px = 1
    if direction.count('y') > 0:
        if side.count('m') > 0:
            my = 1
        if side.count('p') > 0:
            py = 1

    _cut_field.data = Field.data[my:ny-py, mx:nx-px]
    _cut_field.x = _cut_field.x[mx:nx-px]
    _cut_field.y = _cut_field.y[my:ny-py]

    return _cut_field

def vector_field(vx, vy, **karg):
    nsx = nsy = 3
    T,R = np.meshgrid(vx.x[::nsx],vx.y[::nsy])
    X = R * np.cos(T)
    Y = R * np.sin(T)
    vx = vx.data[::nsy,::nsx]
    vy = vy.data[::nsy,::nsx]
    U = vy*np.cos(T) - vx*np.sin(T)
    V = vy*np.sin(T) + vx*np.cos(T)
    ax = plt.gca()
    ax.quiver(X, Y, U, V, scale=3, pivot='middle', **karg)

def vortensity(rho, vx, vy):
    """
    Computing the vortensity from a staggered velocity field.
    """
    Tx, Rx = np.meshgrid(vx.x, vx.y)
    Ty, Ry = np.meshgrid(vy.x, vy.y)

    rvx = Rx * (vx.data)

    curldata = (rvx[1:,1:] - rvx[:-1,1:]) / (Rx[1:,1:] - Rx[:-1,1:]) - \
                (vy.data[1:,1:] - vy.data[1:,:-1]) / (Ty[1:,1:] - Ty[1:,:-1])

    curl = copy.deepcopy(vx)
    curl.nx = curl.nx - 1
    curl.ny = curl.ny - 1
    curl.x    = vx.x[:-1]
    curl.y    = vy.y[:-1]

    rho_corner = 0.25 * (rho.data[1:,1:] + rho.data[:-1,:-1] + \
                       rho.data[1:,:-1] + rho.data[:-1,1:])

    T,R = np.meshgrid(curl.x, curl.y)
    curl.data = curldata / R
    curl.data = (curl.data + 2.0 * rho.omegaframe) / rho_corner

    return curl


def bilinear(x,y,f,p):
    """
    Bilinear interpolation.
    Parameters
    ----------
    x = (x1,x2); y = (y1,y2)
    f = (f11,f12,f21,f22)
    p = (x,y)
    where x,y are the interpolated points and
    fij are the values of the function at the
    points (xi,yj).
    Output
    ------
    f(p): Float.
          The interpolated value of the function f(p) = f(x,y)
    """

    xp  = p[0]; yp   = p[1]; x1  = x[0]; x2  = x[1]
    y1  = y[0]; y2  = y[1];  f11 = f[0]; f12 = f[1]
    f21 = f[2]; f22 = f[3]
    t = (xp-x1)/(x2-x1);    u = (yp-y1)/(y2-y1)

    return (1.0-t)*(1.0-u)*f11 + t*(1.0-u)*f12 + t*u*f22 + u*(1-t)*f21

def get_v(v, x, y):
    """
    For a real set of coordinates (x,y), returns the bilinear
    interpolated value of a Field class.
    """

    i = int((x-v.xmin)/(v.xmax-v.xmin)*v.nx)
    j = int((y-v.ymin)/(v.ymax-v.ymin)*v.ny)

    if i<0 or j<0 or i>v.data.shape[1]-2 or j>v.data.shape[0]-2:
        return None

    f11 = v.data[j,i]
    f12 = v.data[j,i+1]
    f21 = v.data[j+1,i]
    f22 = v.data[j+1,i+1]
    try:
        x1  = v.x[i]
        x2  = v.x[i+1]
        y1  = v.y[j]
        y2  = v.y[j+1]
        return bilinear((x1,x2),(y1,y2),(f11,f12,f21,f22),(x,y))
    except IndexError:
        return None


def transformVelocityFieldToCartesian(vx, vy):
    x = np.tile(vx.x, (len(vx.y), 1))

    vx_t = copy.deepcopy(vx)
    vy_t = copy.deepcopy(vy)

    vy_t.data = -vy.data * np.sin(x) - vx.data * np.cos(x)
    vx_t.data = -vy.data * np.cos(x) + vx.data * np.sin(x)
    return vx_t, vy_t

def polarCoordsToCartesian(field):
    T, R = np.meshgrid(field.x, field.y)
    X = R * np.cos(T)
    Y = R * np.sin(T)
    return X, Y

def interpolateToUniformGrid(field, x_range, y_range):
    newField = copy.deepcopy(field)
    x, y = polarCoordsToCartesian(newField)
    x = np.ravel(x)
    y = np.ravel(y)
    data = np.ravel(newField.data)
    points = np.column_stack((x, y))
    x_space = np.linspace(*x_range)
    y_space = np.linspace(*y_range)

    grid_x, grid_y = np.meshgrid(np.linspace(*x_range), np.linspace(*y_range))
    newData = scipy.interpolate.griddata(points, data, (grid_x, grid_y))
    grid_r = np.sqrt(grid_x**2 + grid_y**2)
    newData[grid_r < np.min(field.y)] = np.nan
    newField.data = newData
    newField.x = x_space
    newField.y = y_space
    return newField
