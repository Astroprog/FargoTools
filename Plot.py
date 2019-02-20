import copy
import numpy as np
import matplotlib.pyplot as plt
from .Tools import Tools

class DiskPlot:
    @staticmethod
    def plotScalar(field, log=False, cartesian=False, cmap='magma', scale=5.2, **karg):
        """
        A layer for plt.imshow or pcolormesh function.
        if cartesian = True, pcolormesh is launched.
        """
        ax = plt.gca()
        if log:
            data = np.log10(field.data)
        else:
            data = field.data
        
        if cartesian:
            interpolated = 0.5 * (data[:,0] + data[:,-1])
            data = np.c_[data, interpolated]
            x = copy.deepcopy(field.x)
            x = np.append(x, x[-1] + (x[-1] - x[-2]))
            T,R = np.meshgrid(x, field.y)
            X = R * np.cos(T)
            Y = R * np.sin(T)
            X *= scale
            Y *= scale
            plt.pcolormesh(X, Y, data, cmap=cmap, **karg)
        else:
            T,R = np.meshgrid(field.x, field.y)
            plt.pcolormesh(T, R, data, cmap=cmap, **karg)
            #ax.imshow(data, cmap=cmap, origin='lower', aspect='auto',
            #          extent=[self.x[0], self.x[-1], self.y[0], self.y[-1]],
            #          **karg)

    #The same but for contours
    @staticmethod
    def plotContours(field, log=False, cartesian=False, scale=5.2, **karg):
        if log:
            data = np.log10(field.data)
        else:
            data = field.data
        ax = plt.gca()
        T,R = np.meshgrid(field.x, field.y)
        if cartesian:
            X = R * np.cos(T)
            Y = R * np.sin(T)
            X *= scale
            Y *= scale
            plt.contour(X, Y, data, **karg)
        else:
            plt.contour(T, R, data, **karg)
    
    @staticmethod
    def plotVectorField(data, fluid=0, scale=5.2, **karg):
        nsx = nsy = 10
        vx = data.fluids[fluid]['vx']
        vy = data.fluids[fluid]['vy']
        vxc = Tools.shift_field(vx, 'x')
        vyc = Tools.shift_field(vy, 'y')
        vxcc = Tools.cut_field(vxc, direction='y', side='p')
        vycc = Tools.cut_field(vyc, direction='x', side='p')

        T,R = np.meshgrid(vxcc.x[::nsx],vxcc.y[::nsy])
        X = R * np.cos(T)
        Y = R * np.sin(T)
        X *= scale
        Y *= scale
        vxcc = vxcc.data[::nsy,::nsx]
        vycc = vycc.data[::nsy,::nsx]
        U = vycc*np.cos(T) - vxcc*np.sin(T)
        V = vycc*np.sin(T) + vxcc*np.cos(T)
        ax = plt.gca()
        ax.quiver(X, Y, U, V, scale=2, pivot='middle', **karg)

    @staticmethod
    def plotStreamLines(data, fluid=0, scale=5.2, density=3, linewidth=1,
                        range_x=[-50, 50, 1000], range_y=[-50, 50, 1000], **karg):
        
        vx = data.fluids[fluid]['vx']
        vy = data.fluids[fluid]['vy']
        vx_t, vy_t = Tools.transformVelocityFieldToCartesian(vx, vy)
        vx_i = Tools.interpolateToUniformGrid(vx_t, range_x, range_y)
        vy_i = Tools.interpolateToUniformGrid(vy_t, range_x, range_y)
        plt.streamplot(vx_i.x * scale, vx_i.y * scale, vx_i.data, vy_i.data, density=3, linewidth=1, **karg)






