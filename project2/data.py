import os
import time
import scipy.io
import numpy as np


def import_data(mat_file):
    """ Import the given matlab data file and return the corresponding data matrix """
    start = time.time()
    mat = scipy.io.loadmat(mat_file)
    end = time.time()
    print('Imported data in: %.3f seconds' % (end-start))
    # Return each array of data (radius and z-axis value of pointer, time window and raw measurements)
    r = np.asarray(mat['r_arr'])
    z = np.asarray(mat['z_arr'])
    t = np.asarray(mat['t_window'])
    d = np.asarray(mat['brt_arr'])
    return r, z, t, d


def save_csv_data(r_arr, z_arr, t_window, brt_arr, path):
    """Save data into 4 separate csv files"""
    if not os.path.exists(path):
        # Create a directory to save the csv files in
        try:
            os.mkdir(path)
        except OSError:
            print('Unexpected failure in creation of directory %s' % path)
    start = time.time()
    np.savetxt(path + '/radius_data.csv', r_arr, delimiter=',')
    np.savetxt(path + '/z-axis_data.csv', z_arr, delimiter=',')
    np.savetxt(path + '/twindow_data.csv', t_window, delimiter=',')
    # 3D file saves are not allowed so split the array into individual 2d arrays
    for slice_2d in brt_arr:
        np.savetxt(path + '/raw_data.csv', slice_2d, delimiter=',')
    end = time.time()
    print('Data saved correctly in: %.3f seconds' % (end-start))


if __name__ == '__main__':
    r_arr, z_arr, t_window, brt_arr = import_data('62529_mat_export_1_1.1.mat')
    save_csv_data(r_arr, z_arr, t_window, brt_arr, 'data')
