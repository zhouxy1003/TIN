import numpy as np


class DataInput:
    def __init__(self, data, batch_size):

        self.batch_size = batch_size
        self.data = data
        self.epoch_size = len(self.data) // self.batch_size
        if self.epoch_size * self.batch_size < len(self.data):
            self.epoch_size += 1
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):

        if self.i == self.epoch_size:
            raise StopIteration

        ts = self.data[self.i * self.batch_size: min((self.i + 1) * self.batch_size,
                                                     len(self.data))]
        self.i += 1

        u, i, y, sl, pos, sl_new = [], [], [], [], [], []
        for t in ts:
            u.append(t[0])
            i.append(t[2])
            y.append(t[3])
            sl.append(len(t[1]))
            pos.append(t[5])
            sl_new.append(len(t[1]) + 1)
        max_sl = max(sl)
        max_sl_new = max(sl_new)

        hist_i = np.zeros([len(ts), max_sl], np.int64)
        pos_i = np.zeros([len(ts), max_sl], np.int64)
        pos_new = np.zeros([len(ts), max_sl_new], np.int64)
        i_new = np.zeros([len(ts), max_sl_new], np.int64)
        k = 0
        for t in ts:
            for l in range(len(t[1])):
                hist_i[k][l] = t[1][l]
                i_new[k][l] = t[1][l]
                pos_i[k][l] = t[4][l]
                pos_new[k][l] = t[4][l]
            pos_new[k][len(t[1])] = 0
            i_new[k][len(t[1])] = t[2]
            k += 1

        return self.i, (u, i, y, hist_i, sl, pos_i, pos, pos_new, i_new, sl_new)


class DataInputTest:
    def __init__(self, data, batch_size):

        self.batch_size = batch_size
        self.data = data
        self.epoch_size = len(self.data) // self.batch_size
        if self.epoch_size * self.batch_size < len(self.data):
            self.epoch_size += 1
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):

        if self.i == self.epoch_size:
            raise StopIteration

        ts = self.data[self.i * self.batch_size: min((self.i + 1) * self.batch_size,
                                                     len(self.data))]
        self.i += 1

        u, i, j, sl, pos, sl_new = [], [], [], [], [], []
        for t in ts:
            u.append(t[0])
            i.append(t[2][0])
            j.append(t[2][1])
            sl.append(len(t[1]))
            pos.append(t[4])
            sl_new.append(len(t[1]) + 1)
        max_sl = max(sl)
        max_sl_new = max(sl_new)

        hist_i = np.zeros([len(ts), max_sl], np.int64)
        pos_i = np.zeros([len(ts), max_sl], np.int64)
        pos_new = np.zeros([len(ts), max_sl_new], np.int64)
        i_new = np.zeros([len(ts), max_sl_new], np.int64)
        j_new = np.zeros([len(ts), max_sl_new], np.int64)

        k = 0
        for t in ts:
            for l in range(len(t[1])):
                hist_i[k][l] = t[1][l]
                i_new[k][l] = t[1][l]
                j_new[k][l] = t[1][l]
                pos_i[k][l] = t[3][l]
                pos_new[k][l] = t[3][l]
            pos_new[k][len(t[1])] = 0
            i_new[k][len(t[1])] = t[2][0]
            j_new[k][len(t[1])] = t[2][1]
            k += 1

        return self.i, (u, i, j, hist_i, sl, pos_i, pos, pos_new, i_new, j_new, sl_new)
