from pydrake.all import (RollPitchYaw, RigidTransform, RotationMatrix)
import numpy as np

DEFAULT_WORKSPACE_CENTRE = np.array([0.4, -0.8, 0.0], dtype=float)
DEFAULT_WORKSPACE_Z = 0.03

def ISO2mine_converter():
    #* ISO9283 workspace centre
    centre = np.array([0.515, 0.0, 0.226])
    rpy = RollPitchYaw(np.array([0, 0, -90])*np.pi/180)
    p = np.array([0., -0.2, 0.])
    X_W_centre = RigidTransform(rpy, p)
    centre_new = X_W_centre.multiply(centre)
    return centre_new
    

def split_workspace(centre=None, z=DEFAULT_WORKSPACE_Z):
    if centre is None:
        centre = DEFAULT_WORKSPACE_CENTRE
    centre = np.asarray(centre, dtype=float)
    edge = 0.4 # in m

    #* get four corners of the square workspace
    corner1 = centre + np.array([ edge/2,  edge/2,  0.])
    corner2 = centre + np.array([-edge/2,  edge/2,  0.])
    corner3 = centre + np.array([-edge/2, -edge/2,  0.])
    corner4 = centre + np.array([ edge/2, -edge/2,  0.])

    #* square edges
    edge1 = [corner1, corner2]
    edge2 = [corner2, corner3]
    edge3 = [corner3, corner4]
    edge4 = [corner4, corner1]

    #* split into 9 blocks on the table and compute its centre coordinates
    d = edge / 3.
    p1 = centre + np.array([ d,-d, 0])
    p2 = centre + np.array([ 0,-d, 0])
    p3 = centre + np.array([-d,-d, 0])
    p4 = centre + np.array([ d, 0, 0])
    p5 = centre + np.array([ 0, 0, 0])
    p6 = centre + np.array([-d, 0, 0])
    p7 = centre + np.array([ d, d, 0])
    p8 = centre + np.array([ 0, d, 0])
    p9 = centre + np.array([-d, d, 0])

    #* grid vertices
    line1_end_pts = [centre + np.array([d/2., -d*(3./2.), 0.]), 
                     centre + np.array([d/2., d*(3/2.), 0.])]

    line2_end_pts = [centre + np.array([-d/2., -d*(3./2.), 0.]), 
                     centre + np.array([-d/2., d*(3/2.), 0.])]

    line3_end_pts = [centre + np.array([-d*(3./2.), d/2., 0.]), 
                     centre + np.array([d*(3./2.), d/2., 0.])]

    line4_end_pts = [centre + np.array([-d*(3./2.), -d/2., 0.]), 
                     centre + np.array([d*(3./2.), -d/2., 0.])]

    #* dict to store coords
    coords = dict()
    coords["corners"] = [corner1, corner2, corner3, corner4]
    coords["block_centres"] = [p1, p2, p3, p4, p5, p6, p7, p8, p9]
    coords["grid_vertices"] = [line1_end_pts, line2_end_pts, line3_end_pts, line4_end_pts]
    coords["edges"] = [edge1, edge2, edge3, edge4]

    return coords





if __name__ == "__main__":
    ISO2mine_converter()
