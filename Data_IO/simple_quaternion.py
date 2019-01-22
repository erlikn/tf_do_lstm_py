import numpy as np

def euler_2_quaternion(yaw, pitch, roll):
    '''
    Input:
        yaw (Z), pitch (Y), roll (X)
    Return:
        q -> [w, x, y, z]
    '''
    # Abbreviations for the various angular functions
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    q = np.zeros([4])
    q[0] = cy * cp * cr + sy * sp * sr # q.w()
    q[1] = cy * cp * sr - sy * sp * cr # q.x()
    q[2] = sy * cp * sr + cy * sp * cr # q.y()
    q[3] = sy * cp * cr - cy * sp * sr # q.z()
    return q


def quaternion_2_euler(q):
    '''
    Input:
        q -> [w, x, y, z]
    Return:
        yaw (Z), pitch (Y), roll (X)
    '''
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[0] * q[1] + q[2] * q[3])
    cosr_cosp = 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2])
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    # pitch (y-axis rotation)
    sinp = 2.0 * (q[0] * q[2] - q[3] * q[1])
    if (np.abs(sinp) >= 1): # use 90 degrees if out of range
        if sinp >= 0:
            pitch = np.pi/2 
        else:
            pitch = -(np.pi/2) 		
    else:
	    pitch = np.arcsin(sinp)
    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[0] * q[3] + q[1] * q[2])
    cosy_cosp = 1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3])
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return yaw+np.pi, pitch+np.pi, roll+np.pi

'''
import Data_IO.simple_quaternion as pyquat
import numpy as np
x=[-0.5, -3.13, 1]
x=[0.5, -3.13, -1]
q = pyquat.euler_2_quaternion(x[0],x[1],x[2])
x
pyquat.quaternion_2_euler(q)
np.sin(x)
np.sin(pyquat.quaternion_2_euler(q))
np.cos(x)
np.cos(pyquat.quaternion_2_euler(q))
'''

def _get_rotmat_from_abg(abg):
    """
    abgxyz is a 6 valued vector: Alpha_yaw, Beta_pitch, Gamma_roll, DeltaX, DeltaY, DeltaZ
    Output is a 3x4 tmat
    For rotation matrix:
    http://planning.cs.uiuc.edu/node102.html
    For Translation side:
    dX, dY, dZ are last column
    """
    a = abg[0]/2
    b = abg[1]/2
    g = abg[2]/2
    tmat = np.array([
              [np.cos(a)*np.cos(b), (np.cos(a)*np.sin(b)*np.sin(g))-(np.sin(a)*np.cos(g)), (np.cos(a)*np.sin(b)*np.cos(g))+(np.sin(a)*np.sin(g))],
              [np.sin(a)*np.cos(b), (np.sin(a)*np.sin(b)*np.sin(g))+(np.cos(a)*np.cos(g)), (np.sin(a)*np.sin(b)*np.cos(g))-(np.cos(a)*np.sin(g))],
              [-np.sin(b),          np.cos(b)*np.sin(g),                                   np.cos(b)*np.cos(g),                                 ]
           ], dtype=np.float32)
    return tmat

def plot():
    from matplotlib import pyplot
    x = np.arange(-(2*np.pi), 2*np.pi, 0.01)
    res = list()
    res.append(list())
    res.append(list())
    res.append(list())
    for i in range(3):
        for j in range(x.shape[0]):
            if i == 0:
                z =[x, 0, 0]
            elif i == 1:
                z =[0, x, 0]
            else:
                z =[0, 0, x]
            res[i].append(quaternion_2_euler(euler_2_quaternion(z[0], z[1], z[2])))
            

