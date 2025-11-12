"""
Cloud in cell section, to put profiles on grid by accounting for the shift of halo position with respect to pixels centers.
"""
import numpy as np
from astropy.convolution import convolve_fft
from .parameters import Parameters

def CIC_coefficients(parameters: Parameters, H_X, H_Y, H_Z):
    """
    The 27 CIC coefficients correspond to : (center), (xVol_left ,xVol_right,yVol_left ,yVol_right,zVol_left ,zVol_right),
    (xy_side11  ,xy_side_11 ,xy_side1_1 ,xy_side_1_1)
    (yz_side11 ,yz_side_11,yz_side1_1,yz_side_1_1 )
    (xz_side11 ,xz_side_11 ,xz_side1_1 ,xz_side_1_1)
    (corner111  ,corner1_11 ,corner11_1 ,corner1_1_1,corner_111 ,corner_1_11,corner_11_1,corner_1_1_1)
    --> central weight, 6 nearest pixels to the central one, then the 12 closest one, finally the 8 corners of the cube

    Description of the function : given halo positions H_X, H_Y, H_Z, instead of assuming halos live a the center of the grid cell,
                we attribute weight to the central + 26 closest pixels to the halo center. We do this according to the cloud-in-cell interpolation.
                It seems not to be changing anything at the GS, PS level.. Order 1%, to be double checked..

    Parameters
    ----------
    H_X, H_Y, H_Z : real space positions of halos (between 0 and Lbox)
    param : dictionnary containing all the input parameters

    Returns
    -------
    pixel_coordinates_output, cic_coeff_output : two arrays of same size.
        pixel... :  pixel coordinates. Will paint a profile around it.
        cic...   : the coefficient to assign to this pixel when painting the profile

    """

    LBox = parameters.simulation.Lbox  # Mpc/h
    nGrid = parameters.simulation.Ncell  # number of grid cells
    pix = LBox / nGrid  # pixel real size

    ### shift the positions of the halos to the closest point among the 27 edges, vertices and center of the voxel.
    ### this is to make the cic procedure faster : in forces to have limited unique number of cic coeff.
    H_X,H_Y,H_Z = np.round(H_X * nGrid * 2 / LBox) / (2 * nGrid) * LBox,np.round(H_Y * nGrid * 2 / LBox) / (2 * nGrid) * LBox, np.round(H_Z * nGrid * 2 / LBox) / (2 * nGrid) * LBox

    # positions in pixel units
    X_pix = np.array([H_X / LBox * nGrid]).astype(int)[0]
    Y_pix = np.array([H_Y / LBox * nGrid]).astype(int)[0]
    Z_pix = np.array([H_Z / LBox * nGrid]).astype(int)[0]

    ## real space coordinate of the boundaries of the pixel the halo lies in
    up_X, up_Y, up_Z = LBox / nGrid * (np.array([np.array((H_X, H_Y, H_Z)) / LBox * nGrid]).astype(int)[0] + 1)
    down_X, down_Y, down_Z = LBox / nGrid * np.array([np.array((H_X, H_Y, H_Z)) / LBox * nGrid]).astype(int)[0]

    x_1 = np.abs(np.minimum(0,
                            H_X - pix / 2 - down_X))  # distance between lower X edge of cell center on halo with lower X edge of the grid pixel
    x1 = np.maximum(0, H_X + pix / 2 - up_X)
    x0 = pix - x1 - x_1

    y_1 = np.abs(np.minimum(0,
                            H_Y - pix / 2 - down_Y))  # distance between lower Y edge of cell center on halo with lower Y edge of the grid pixel
    y1 = np.maximum(0, H_Y + pix / 2 - up_Y)
    y0 = pix - y1 - y_1

    z_1 = np.abs(np.minimum(0,
                            H_Z - pix / 2 - down_Z))  # distance between lower Z edge of cell center on halo with lower Z edge of the grid pixel
    z1 = np.maximum(0, H_Z + pix / 2 - up_Z)
    z0 = pix - z1 - z_1

    # edges to the cube center
    xVol_left = x_1 * y0 * z0
    xVol_right = x1 * y0 * z0

    yVol_left = y_1 * x0 * z0
    yVol_right = y1 * x0 * z0

    zVol_left = z_1 * y0 * x0
    zVol_right = z1 * y0 * x0

    # cube center
    centerVol = x0 * y0 * z0

    # corner from plane xy
    xy_side11 = z0 * x1 * y1
    xy_side_11 = z0 * x_1 * y1
    xy_side1_1 = z0 * x1 * y_1
    xy_side_1_1 = z0 * x_1 * y_1

    # corner from plane yz
    yz_side11 = x0 * y1 * z1
    yz_side_11 = x0 * y_1 * z1
    yz_side1_1 = x0 * y1 * z_1
    yz_side_1_1 = x0 * y_1 * z_1

    # corner from plane xz
    xz_side11 = y0 * x1 * z1
    xz_side_11 = y0 * x_1 * z1
    xz_side1_1 = y0 * x1 * z_1
    xz_side_1_1 = y0 * x_1 * z_1

    # 8 corners of the cube
    corner111 = x1 * y1 * z1
    corner1_11 = x1 * y_1 * z1
    corner11_1 = x1 * y1 * z_1
    corner1_1_1 = x1 * y_1 * z_1
    corner_111 = x_1 * y1 * z1
    corner_1_11 = x_1 * y_1 * z1
    corner_11_1 = x_1 * y1 * z_1
    corner_1_1_1 = x_1 * y_1 * z_1

    ## for one halo, np.sum(output) = pixel_volume (i.e pix**3)
    ## but in total we should assign a weight 1 spread over the 27 pixels.
    ## that's why renormalize (by dividing by pix**3)
    output = np.array((centerVol, xVol_left, xVol_right,
                       yVol_left, yVol_right, zVol_left,
                       zVol_right, xy_side11, xy_side_11,
                       xy_side1_1, xy_side_1_1, yz_side11,
                       yz_side_11, yz_side1_1, yz_side_1_1,
                       xz_side11, xz_side_11, xz_side1_1,
                       xz_side_1_1, corner111, corner1_11,
                       corner11_1, corner1_1_1, corner_111,
                       corner_1_11, corner_11_1, corner_1_1_1)).T/pix**3

    # pixel coordinates (between 0 and nGrid) corresponding to the CIC coefficients
    # shape if (27,len(H_X))
    pixel_coordinates_Z = np.array([Z_pix, Z_pix, Z_pix,
                                    Z_pix, Z_pix, Z_pix - 1,
                                    Z_pix + 1, Z_pix, Z_pix,
                                    Z_pix, Z_pix, Z_pix + 1,
                                    Z_pix + 1, Z_pix - 1, Z_pix - 1,
                                    Z_pix + 1, Z_pix + 1, Z_pix - 1,
                                    Z_pix - 1, Z_pix + 1, Z_pix + 1,
                                    Z_pix - 1, Z_pix - 1, Z_pix + 1,
                                    Z_pix + 1, Z_pix - 1, Z_pix - 1])

    pixel_coordinates_Y = np.array([Y_pix, Y_pix, Y_pix,
                                    Y_pix - 1, Y_pix + 1, Y_pix,
                                    Y_pix, Y_pix + 1, Y_pix + 1,
                                    Y_pix - 1, Y_pix - 1, Y_pix + 1,
                                    Y_pix - 1, Y_pix + 1, Y_pix - 1,
                                    Y_pix, Y_pix, Y_pix,
                                    Y_pix, Y_pix + 1, Y_pix - 1,
                                    Y_pix + 1, Y_pix - 1, Y_pix + 1,
                                    Y_pix - 1, Y_pix + 1, Y_pix - 1])

    pixel_coordinates_X = np.array([X_pix, X_pix - 1, X_pix + 1,
                                    X_pix, X_pix, X_pix,
                                    X_pix, X_pix + 1, X_pix - 1,
                                    X_pix + 1, X_pix - 1, X_pix,
                                    X_pix, X_pix, X_pix,
                                    X_pix + 1, X_pix - 1, X_pix + 1,
                                    X_pix - 1, X_pix + 1, X_pix + 1,
                                    X_pix + 1, X_pix + 1, X_pix - 1,
                                    X_pix - 1, X_pix - 1, X_pix - 1])

    ## flattened array with all cic coeff (size is 27*len(H_X))
    cic_coeff_output = output.flatten()
    non_zero_index = np.where(cic_coeff_output > 0)
    cic_coeff_output = cic_coeff_output[non_zero_index]
    ## flattened array with all cic pixels coordinates,  (size is (3,27*len(H_X)))
    pixel_coordinates_output = np.array((pixel_coordinates_X.T.flatten()[non_zero_index] % nGrid,
                                         pixel_coordinates_Y.T.flatten()[non_zero_index] % nGrid,
                                         pixel_coordinates_Z.T.flatten()[non_zero_index] % nGrid))
    if not round(np.sum(cic_coeff_output), 6) == len(H_X):
        print('ERROR IN CIC ASSIGNATION IN CIC_coefficients, in cloud_in_cells.py')
        exit()


    return pixel_coordinates_output, cic_coeff_output





def put_profiles_group_CIC(source_pos, nbr_of_halos, coef_cic, profile_kern, nGrid=None):
    '''
    ---- THIS IS WRONG ----
    Profiles with cloud in cell technique.
    coef = 27 elements with CIC coefficients

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

