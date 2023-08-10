import numpy as np

sess_nums = 10


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
        # (115053, [2472, 12254, 40489, 40491, 42250, 42264, 7681], 29205, 1, [[2472, 12254, 40489, 40491, 42250,
        # 42264], [7681], [], [], [], [], [], [], [], []], [[2, 2, 2, 2, 2, 2], [1]], [[6, 5, 4, 3, 2, 1], [1]])
        u, i, y, sl = [], [], [], []
        sess_sl = np.zeros([len(ts), sess_nums])
        s_i = 0
        for t in ts:
            u.append(t[0])
            i.append(t[2])
            y.append(t[3])
            sl.append(len(t[5]))  # number of sessions
            for n in range(len(t[5])):
                sess_sl[s_i][n] = len(t[5][n])
            s_i += 1
        max_sess_sl = int(np.max(sess_sl))
        sess_i = np.zeros([len(ts), sess_nums, max_sess_sl], np.int64)
        pos_i_k = np.zeros([len(ts), sess_nums, max_sess_sl], np.int64)
        pos_i_t = np.zeros([len(ts), sess_nums, max_sess_sl], np.int64)

        k = 0
        for t in ts:
            for n in range(len(t[5])):
                for m in range(len(t[5][n])):
                    sess_i[k][n][m] = t[4][n][m]
                    pos_i_k[k][n][m] = t[5][n][m]
                    pos_i_t[k][n][m] = t[6][n][m]
            k += 1

        return self.i, (u, i, y, sess_i, sl, pos_i_k, pos_i_t, sess_sl)


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

        # (180873, [4763, 29004, 24347, 20887], (37639, 39715), [[4763, 29004], [24347], [20887], [], [], [], [], [],
        # [], []], [[3, 3], [2], [1]], [[2, 1], [1], [1]])
        u, i, j, sl = [], [], [], []
        sess_sl = np.zeros([len(ts), sess_nums])
        s_i = 0
        for t in ts:
            u.append(t[0])
            i.append(t[2][0])
            j.append(t[2][1])
            sl.append(len(t[4]))
            for n in range(len(t[4])):
                sess_sl[s_i][n] = len(t[4][n])
            s_i += 1
        max_sess_sl = int(np.max(sess_sl))
        # print(max_sess_sl)
        sess_i = np.zeros([len(ts), sess_nums, max_sess_sl], np.int64)
        pos_i_k = np.zeros([len(ts), sess_nums, max_sess_sl], np.int64)
        pos_i_t = np.zeros([len(ts), sess_nums, max_sess_sl], np.int64)
        k = 0
        for t in ts:
            for n in range(len(t[4])):
                for m in range(len(t[4][n])):
                    sess_i[k][n][m] = t[3][n][m]
                    pos_i_k[k][n][m] = t[4][n][m]
                    pos_i_t[k][n][m] = t[5][n][m]
            k += 1

        return self.i, (u, i, j, sess_i, sl, pos_i_k, pos_i_t, sess_sl)
