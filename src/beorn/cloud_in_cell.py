"""
Cloud in cell section, to put profiles on grid by accounting for the shift of halo position with respect to pixels centers.
"""
import numpy as np




def put_profiles_group_CIC(source_pos, nbr_of_halos, coef_cic, profile_kern, nGrid=None):
    '''
    Profiles with cloud in cell technique.
    coef = 27 elements with CIC coefficients
    structure of coef : (center), (xVol_left ,xVol_right,yVol_left ,yVol_right,zVol_left ,zVol_right),
    (xy_side11  ,xy_side_11 ,xy_side1_1 ,xy_side_1_1)
    (yz_side11 ,yz_side_11,yz_side1_1,yz_side_1_1 )
    (xz_side11 ,xz_side_11 ,xz_side1_1 ,xz_side_1_1)
    (corner111  ,corner1_11 ,corner11_1 ,corner1_1_1,corner_111 ,corner_1_11,corner_11_1,corner_1_1_1)
    --> central weight, 6 nearest pixels to the central one, then the 12 closest one, finally the 8 corners of the cube

    source_pos : the position of halo centered in units number of grid cell (0..nGrid-1). shape is (3,N), with N the number of halos. (X,Y,Z)
    nbr_of_halos : the number of halos in each source_pos (>0), array of size len(source_pos)
    profile_kern : the  profile to put on a grid around each source_pos pixels, multiplied by nbr_of_halos.

    Bin halos masses to do this. Then in a given bin all halos are assumed to have the same profile. This speeds up dramatically this step.
    '''

    if nGrid is None: nGrid = profile_kern.shape[0]
    source_grid = np.zeros((nGrid, nGrid, nGrid))

    for ii in range(len(source_pos[0])):
        i, j, k = source_pos[0][ii], source_pos[1][ii], source_pos[2][ii]
        nbr_h = nbr_of_halos[ii]
        coef = coef_cic[ii]
        print('sum of coef is ', np.sum(coef))

        source_grid[i, j, k] += nbr_h * coef[0]

        source_grid[i - 1, j, k] += nbr_h * coef[1]
        source_grid[i + 1, j, k] += nbr_h * coef[2]
        source_grid[i, j - 1, k] += nbr_h * coef[3]
        source_grid[i, j + 1, k] += nbr_h * coef[4]
        source_grid[i, j, k - 1] += nbr_h * coef[5]
        source_grid[i, j, k + 1] += nbr_h * coef[6]

        source_grid[i + 1, j + 1, k] += nbr_h * coef[7]
        source_grid[i - 1, j + 1, k] += nbr_h * coef[8]
        source_grid[i + 1, j - 1, k] += nbr_h * coef[9]
        source_grid[i - 1, j - 1, k] += nbr_h * coef[10]

        source_grid[i, j + 1, k + 1] += nbr_h * coef[11]
        source_grid[i, j - 1, k + 1] += nbr_h * coef[12]
        source_grid[i, j + 1, k - 1] += nbr_h * coef[13]
        source_grid[i, j - 1, k - 1] += nbr_h * coef[14]

        source_grid[i + 1, j, k + 1] += nbr_h * coef[15]
        source_grid[i - 1, j, k + 1] += nbr_h * coef[16]
        source_grid[i + 1, j, k - 1] += nbr_h * coef[17]
        source_grid[i - 1, j, k - 1] += nbr_h * coef[18]

        source_grid[i + 1, j + 1, k + 1] += nbr_h * coef[19]
        source_grid[i + 1, j - 1, k + 1] += nbr_h * coef[20]
        source_grid[i + 1, j + 1, k - 1] += nbr_h * coef[21]
        source_grid[i + 1, j - 1, k - 1] += nbr_h * coef[22]
        source_grid[i - 1, j + 1, k + 1] += nbr_h * coef[23]
        source_grid[i - 1, j - 1, k + 1] += nbr_h * coef[24]
        source_grid[i - 1, j + 1, k - 1] += nbr_h * coef[25]
        source_grid[i - 1, j - 1, k - 1] += nbr_h * coef[26]

    out = convolve_fft(source_grid, profile_kern, boundary='wrap', normalize_kernel=False, allow_huge=True)
    return out



def CIC_coefficients(param,H_X, H_Y, H_Z):

    LBox = param.sim.Lbox  # Mpc/h
    nGrid = param.sim.Ncell  # number of grid cells
    pix = LBox / nGrid # pixel size

    up_X, up_Y, up_Z = LBox / nGrid * (np.array([(H_X, H_Y, H_Z) / LBox * nGrid]).astype(int)[0] + 1)
    down_X, down_Y, down_Z = LBox / nGrid * np.array([(H_X, H_Y, H_Z) / LBox * nGrid]).astype(int)[0]

    x_1 = np.abs(np.minimum(0,X - pix / 2 - down_X))  # distance between lower X edge of cell center on halo with lower X edge of the grid pixel
    x1 = np.maximum(0, X + pix / 2 - up_X)
    x0 = pix - x1 - x_1

    y_1 = np.abs(np.minimum(0,Y - pix / 2 - down_Y))  # distance between lower Y edge of cell center on halo with lower Y edge of the grid pixel
    y1 = np.maximum(0, Y + pix / 2 - up_Y)
    y0 = pix - y1 - y_1

    z_1 = np.abs(np.minimum(0, Z - pix / 2 - down_Z))  # distance between lower Z edge of cell center on halo with lower Z edge of the grid pixel
    z1 = np.maximum(0, Z + pix / 2 - up_Z)
    z0 = pix - z1 - z_1

    #edges to the cube center
    xVol_left  = x_1*y0*z0
    xVol_right = x1*y0*z0

    yVol_left  = y_1*x0*z0
    yVol_right = y1*x0*z0

    zVol_left  = z_1*y0*z0
    zVol_right = z1*y0*z0

    #cube center
    centerVol= x0*y0*z0

    #corner from plane xy
    xy_side11  = z0*x1*y1
    xy_side_11 = z0*x_1*y1
    xy_side1_1 = z0*x1*y_1
    xy_side_1_1 = z0*x_1*y_1

    #corner from plane yz
    yz_side11 = x0*y1*z1
    yz_side_11 = x0*y_1*z1
    yz_side1_1 = x0*y1*z_1
    yz_side_1_1 = x0*y_1*z_1

    #corner from plane xz
    xz_side11 = y0*x1*z1
    xz_side_11 = y0*x_1*z1
    xz_side1_1 = y0*x1*z_1
    xz_side_1_1 =y0*x_1*z_1

    #8 corners of the cube
    corner111  =  x1*y1*z1
    corner1_11 =  x1*y_1*z1
    corner11_1 =  x1*y1*z_1
    corner1_1_1 = x1*y_1*z_1
    corner_111 =  x_1*y1*z1
    corner_1_11 =  x_1*y_1*z1
    corner_11_1 =  x_1*y1*z_1
    corner_1_1_1 =  x_1*y_1*z_1

    output = np.array((centerVol,xVol_left ,xVol_right,yVol_left ,yVol_right,zVol_left ,zVol_right,xy_side11  ,xy_side_11 ,xy_side1_1 ,xy_side_1_1,yz_side11 ,yz_side_11,yz_side1_1,yz_side_1_1 ,xz_side11 ,xz_side_11 ,xz_side1_1 ,xz_side_1_1,corner111  ,corner1_11 ,corner11_1 ,corner1_1_1,corner_111 ,corner_1_11,corner_11_1,corner_1_1_1)).T

    return output