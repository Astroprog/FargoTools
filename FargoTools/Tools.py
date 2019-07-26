import copy
import numpy as np
import scipy.interpolate
from astropy import constants as const
from astropy import units as u
from astropy.modeling.blackbody import blackbody_lambda, blackbody_nu
from simdata import data, grid

class Tools:
    @staticmethod
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

    @staticmethod
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


    @staticmethod
    def vortensity(data, fluid=0):
        """
        Computing the vortensity from a staggered velocity field.
        """
        rho = data.fluids[fluid]['density']
        vx = data.fluids[fluid]['vx']
        vy = data.fluids[fluid]['vy']
        
        vxc = Tools.shift_field(vx, 'x')
        vyc = Tools.shift_field(vy, 'y')
        vxcc = Tools.cut_field(vxc, direction='y', side='p')
        vycc = Tools.cut_field(vyc, direction='x', side='p')
        rhoc = Tools.cut_field(rho, direction='xy', side='p')

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


    @staticmethod
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

    @staticmethod
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


    @staticmethod
    def transformVelocityFieldToCartesian(vx, vy):
        x = np.tile(vx.x, (len(vx.y), 1))

        vx_t = copy.deepcopy(vx)
        vy_t = copy.deepcopy(vy)

        vy_t.data = -vy.data * np.sin(x) - vx.data * np.cos(x)
        vx_t.data = -vy.data * np.cos(x) + vx.data * np.sin(x)
        return vx_t, vy_t

    @staticmethod
    def polarCoordsToCartesian(field, interfaces=True):
        if interfaces:
            T, R = np.meshgrid(field.grid.get_interfaces("phi"), field.grid.get_interfaces("r"))
        else:
            T, R = np.meshgrid(field.grid.get_centers("phi"), field.grid.get_centers("r"))
        X = R * np.cos(T * u.rad)
        Y = R * np.sin(T * u.rad)
        return X, Y

    @staticmethod
    def interpolateToUniformGrid(field, x_range, y_range):
        newField = copy.deepcopy(field)
        x, y = Tools.polarCoordsToCartesian(newField, interfaces=False)
        x = np.ravel(x)
        y = np.ravel(y)
        data = np.ravel(newField.data)
        points = np.column_stack((x, y))
        x_space = np.linspace(*x_range)
        y_space = np.linspace(*y_range)

        grid_x, grid_y = np.meshgrid(np.linspace(*x_range), np.linspace(*y_range))
        newData = scipy.interpolate.griddata(points, data, (grid_x, grid_y))
        grid_r = np.sqrt(grid_x**2 + grid_y**2)
        newData[grid_r < np.min(field.grid.get_centers("r"))] = np.nan
        newField.data = newData
        newField.grid = grid.RegularGrid(x1_i=x_space, x2_i=y_space)
        return newField

    @staticmethod
    def getOpticalDepth(field, incl=0.0, opacity=3.0*u.cm**2 / u.g, scale=1.0, stellarMass=1.0):
        newField = copy.deepcopy(field)
        rho = newField.data * stellarMass * u.M_sun / (scale * u.AU)**2
        rho = rho.to(u.g / u.cm**2)
        rho *= opacity / np.cos(incl)
        newField.data = rho
        return newField

    @staticmethod
    def getSoundSpeed(aspectRatio, flaringIndex, r, stellarMass=1.0, scale=1.0):
        cs = aspectRatio * r**flaringIndex * np.sqrt(const.GM_sun.cgs * stellarMass / (r * scale * const.au.cgs))
        return cs

    @staticmethod
    def getTemperaturesFargo(aspectRatio, flaringIndex, r, stellarMass=1.0, scale=1.0):
        cs = Tools.getSoundSpeed(aspectRatio, flaringIndex, r, stellarMass=stellarMass, scale=scale)
        return (2.4 * const.m_p.cgs / const.k_B.cgs * cs**2).to(u.K)

    @staticmethod
    def getTemperaturesLuminosity(luminosity, phi, r, scale=1.0):
        return ((phi * luminosity / (8 * np.pi * (r * scale * const.au.cgs)**2 * const.sigma_sb.cgs))**0.25).to(u.K)

    @staticmethod
    def getIntensityMap(nu, tau, T):
        I_nu = copy.deepcopy(tau)
        T = np.tile(T, (tau.data.shape[1], 1)).T
        B_nu = blackbody_nu(nu, T)
        I_nu.data = (1.0 - np.exp(-tau.data)) * B_nu
        return I_nu





