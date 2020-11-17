import time
import scipy.io


def import_data(mat_file):
    """ Import the given matlab data file and return the corresponding data matrix """
    return scipy.io.loadmat(mat_file)


if __name__ == '__main__':
    start = time.time()
    mat = import_data('62529_mat_export_1_1.1.mat')
    end = time.time()
    print('Imported data in: %.3f seconds' % (end-start))

    r_arr = mat['r_arr'] # radius data
    z_arr = mat['z_arr'] # z-axis data
    t_window = mat['t_window'] # time window (1 experiment every 50 ns approx)
    brt_arr = mat['brt_arr'] # 3d array (12*10*200000)