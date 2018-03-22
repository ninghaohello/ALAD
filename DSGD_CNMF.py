import numpy as np
from scipy import sparse as sp
from scipy import io as sio
import multiprocessing as mp
import Mat2BlockPublic as Mat2Block
import math, time, copy, random, sys


class DSGD_CNMF(object):
    def __init__(self, R, B, sp_flag, alpha, gamma1, gamma2, thre, count_exceedThre):
        # hyperparameters
        self.alpha = alpha
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.R = R      # number of latent dims
        self.B = B      # number of blocks for parallel implementation
        self.thre = thre
        self.count_exceedThre = count_exceedThre

        self.sym_flag = 1
        self.shuf_flag = 0
        self.diag_flag = 1
        self.sparse_flag_x = sp_flag

        self.num_node = 0
        self.num_attr = 0


    def ReadInput(self, fn_a, fn_x):
        """A, X: matrices segmented into blocks(bins)
        """
        if fn_a.split('.')[-1] == "csv":
            self.A, self.shuf = Mat2Block.Edges2Bins(fn_a, self.B,
                                                     self.sym_flag,
                                                     self.shuf_flag,
                                                     self.diag_flag)
        elif fn_a.split('.')[-1] == "mat":
            self.A, self.shuf = Mat2Block.Adjmat2Bins(fn_a, self.B,
                                                      self.sym_flag,
                                                      self.shuf_flag,
                                                      self.diag_flag)
        self.X = Mat2Block.X2Bins(fn_x, self.B,
                                  self.shuf_flag, self.shuf,
                                  self.sparse_flag_x)


    def ComputeDimension(self):
        for b in range(self.B):
            self.num_node += self.A[b][0].shape[0]
            self.num_attr += self.X[0][b].shape[1]


    def Update_t(self, Aij, Xb, Wb, Hk, inds, child_pipe):
        """Xb and Wb are lists of two matrices
           Aij and Hk are matrices
        """
        if inds[0] == inds[1]:
            Xik = Xb[0]
            Xjk = Xb[0]
            Wi = Wb[0]
            Wj = Wb[0]
        else:
            Xik, Xjk = Xb
            Wi, Wj = Wb

        # Gradient w.r.t. W^i and/or W^j
        if inds[0] != inds[1]:
            # Off-diagonal Aij
            # Gradient w.r.t. W^i (part 1), for all-zero block of A^{i,j}
            if self.sparse_flag_x == 0:
                Wi_d = 2 * Wi.dot(Wj.T.dot(Wj)) - self.alpha * (Xik - Wi.dot(Hk)).dot(Hk.T) + \
                       (self.gamma1 / self.B) * Wi
            else:
                Wi_d = 2 * Wi.dot(Wj.T.dot(Wj)) + self.alpha * Wi.dot(Hk.dot(Hk.T)) +\
                       (self.gamma1 / self.B) * Wi
                rows, cols = sp.find(Xik)[0:2]
                for e in range(len(rows)):
                    r, c = [int(rows[e]), int(cols[e])]
                    Wi_d[r,:] -= self.alpha * Xik[r,c] * Hk[:,c]

            # Gradient w.r.t. W^j (part 1)
            if self.sparse_flag_x == 0:
                Wj_d = 2 * Wj.dot(Wi.T.dot(Wi)) - self.alpha * (Xjk - Wj.dot(Hk)).dot(Hk.T) + \
                       (self.gamma1/self.B) * Wj
            else:
                Wj_d = 2 * Wj.dot(Wi.T.dot(Wi)) + self.alpha * Wj.dot(Hk.dot(Hk.T)) +\
                       (self.gamma1 / self.B) * Wj
                rows, cols = sp.find(Xjk)[0:2]
                for e in range(len(rows)):
                    r, c = [int(rows[e]), int(cols[e])]
                    Wj_d[r,:] -= self.alpha * Xjk[r,c] * Hk[:,c]

            # Gradient w.r.t. W^i and W^j (part 2), entry-level gradient, for non_zero entries of A^{i,j}
            rows, cols = sp.find(Aij)[0:2]
            for e in range(len(rows)):
                r, c = [int(rows[e]), int(cols[e])]
                Wi_d[r,:] -= 2 * Wj[c,:]
                Wj_d[c,:] -= 2 * Wi[r,:]
        else:
            # Diagonal Aij
            if self.sparse_flag_x == 0:
                Wi_d = 4.*Wi.dot(Wi.T.dot(Wi)) - 2. * (self.alpha * (Xik - Wi.dot(Hk)).dot(Hk.T)) + \
                       2.*(self.gamma1 / self.B) * Wi
            else:
                Wi_d = 4. * Wi.dot(Wi.T.dot(Wi)) + 2. * self.alpha * Wi.dot(Hk.dot(Hk.T)) +\
                       2. * (self.gamma1 / self.B) * Wi
                rows, cols = sp.find(Xik)[0:2]
                for e in range(len(rows)):
                    r, c = [int(rows[e]), int(cols[e])]
                    Wi_d[r,:] -= 2. * self.alpha * Xik[r,c] * Hk[:,c]

            rows, cols = sp.find(Aij)[0:2]
            for e in range(len(rows)):
                r, c = [int(rows[e]), int(cols[e])]
                Wi_d[r,:] -= 4 * Wi[c,:]

        # Gradient w.r.t. H^k
        if self.sparse_flag_x == 0:
            Hk_d = -1*(self.alpha*Wi.T.dot(Xik - Wi.dot(Hk))) - self.alpha*Wj.T.dot(Xjk - Wj.dot(Hk)) + \
                   2.*(self.gamma2/self.B)*Hk
        else:
            Hk_d = self.alpha*Wi.T.dot(Wi).dot(Hk) + self.alpha * (Wj.T.dot(Wj)).dot(Hk) +\
                   2.*(self.gamma2/self.B)*Hk
            if inds[0] == inds[1]:
                rows, cols = sp.find(Xik)[0:2]
                for e in range(len(rows)):
                    r, c = [rows[e], cols[e]]
                    Hk_d[:,c] -= 2. * self.alpha * Xik[r,c] * Wi[r,:]
            else:
                rows, cols = sp.find(Xik)[0:2]
                for e in range(len(rows)):
                    r, c = [rows[e], cols[e]]
                    Hk_d[:,c] -= self.alpha * Xik[r,c] * Wi[r,:]
                rows, cols = sp.find(Xjk)[0:2]
                for e in range(len(rows)):
                    r, c = [rows[e], cols[e]]
                    Hk_d[:,c] -= self.alpha * Xjk[r,c] * Wj[r,:]

        # Put gradients into the output pipe
        if inds[0] != inds[1]:
            child_pipe.send([Wi_d, Wj_d, Hk_d])
        else:
            child_pipe.send([Wi_d, Hk_d])



    def BDSGD_t(self, num_iter, check_iter, epsilon_start, epsilon_end):
        """cost_function = ||A - WW'||^2_f + alpha||X - WH||^2_f + gamma1||W||^2_f
                           + gamma2||H||^2_f
        """
        size_Ab = self.A[0][0].shape
        size_Ab_rb = self.A[-1][-1].shape
        size_Xb = self.X[0][0].shape
        size_Xb_rb = self.X[-1][-1].shape

        epsilon_decay = (epsilon_start - epsilon_end)/num_iter
        B = self.B

        # Initialization W and H(could add energy equilibrium)
        W = [[] for b in range(B)]      # :
        H = [[] for b in range(B)]      # ..
        for b in range(B):
            if b != B-1:
                W[b] = 0.1 * np.random.rand(size_Ab[0], self.R)
                H[b] = 0.1 * np.random.rand(self.R, size_Xb[1])
            else:
                W[b] = 0.1 * np.random.rand(size_Ab_rb[0], self.R)
                H[b] = 0.1 * np.random.rand(self.R, size_Xb_rb[1])

        # BDSGD
        time_start = time.time()

        # check starting error
        print "Time Start"
        err_old = self.CheckError(B, W, H, time_start, time_start)
        count = 0

        epsilon = epsilon_start
        for t in range(num_iter):
            # randomly generate non-repeat triplets for input blocks
            triplets = []
            ind_W = range(B)
            ind_H = range(B)
            while ind_W:
                i = random.sample(ind_W, 1)[0]
                j = random.sample(ind_W, 1)[0]
                k = random.sample(ind_H, 1)[0]
                triplets.append([i,j,k])

                ind_W = [w for w in ind_W if w not in [i,j]]
                ind_H = [h for h in ind_H if h != k]

            procs = [[] for b in range(len(triplets))]
            self.pipes = [mp.Pipe() for b in range(self.B)]
            ## allocate jobs
            for b in range(len(triplets)):
                tplt = triplets[b]
                Ab = self.A[tplt[0]][tplt[1]]
                if tplt[0] == tplt[1]:
                    Xb = [self.X[tplt[0]][tplt[2]]]
                    Wb = [W[tplt[0]]]
                else:
                    Xb = [self.X[tplt[0]][tplt[2]], self.X[tplt[1]][tplt[2]]]
                    Wb = [W[tplt[0]], W[tplt[1]]]
                Hb = H[tplt[2]]
                procs[b] = mp.Process(target=self.Update_t,
                                      args=(Ab, Xb, Wb, Hb, tplt, self.pipes[b][1]))

            for p in procs:
                p.start()

            b = 0
            for p in procs:
                tplt = triplets[b]
                wb_hb = self.pipes[b][0].recv()
                b += 1

                i, j, k = tplt
                if i != j:
                    W[i] -= epsilon * wb_hb[0]
                    W[j] -= epsilon * wb_hb[1]
                    H[k] -= epsilon * wb_hb[2]
                else:
                    W[i] -= epsilon * wb_hb[0]
                    H[k] -= epsilon * wb_hb[1]
                p.join()
            for b in range(self.B):
                W[b] = W[b].clip(min=0)
                H[b] = H[b].clip(min=0)

            # Check approximation error
            if (not t%check_iter) | (t == num_iter-1) | (t in [1,3,10,20]):
                print "Iteration: ", t
                time_inter = time.time()
                print "Time elapsed: ", time_inter - time_start
                err_new = self.CheckError(B, W, H, time_start, time_inter)

                if (err_old - err_new) / err_old < self.thre:
                    count += 1
                    if count >= self.count_exceedThre:
                        break
                else:
                    err_old = err_new

            epsilon -= epsilon_decay

        time_end = time.time()
        print "Time cost: ", time_end - time_start

        self.W = np.vstack(W)
        self.H = np.hstack(H)


    def CheckError(self, B, W, H, time_start, time_inter):
        err_A = 0.
        err_X = 0.
        energy_W = 0.
        energy_H = 0.
        for i in range(B):
            energy_W += np.linalg.norm(W[i]) ** 2
            energy_H += np.linalg.norm(H[i]) ** 2
            for j in range(B):
                err_A += np.linalg.norm(self.A[i][j] - W[i].dot(W[j].T))**2
                err_X += np.linalg.norm(self.X[i][j] - W[i].dot(H[j]))**2
        err = err_A + self.alpha * err_X
        err = err**0.5
        print "Error: ", err

        f = open(fn_x.split('.')[0] + "_" + "log.txt", 'a')
        f.write("Time elapsed: " + str(time_inter - time_start) + '\n' + "Error: " + str(err) + '\n')
        f.close()

        return err


    def SaveWH(self, fn_w, fn_h):
        sio.savemat(fn_w, {'W': self.W})
        sio.savemat(fn_h, {'H': self.H})


    def SaveX(self, fn_x_out):
        for b in range(self.B):
            self.X[b] = np.hstack(self.X[b])
        self.X = np.vstack(self.X)
        sio.savemat(fn_x_out, {'X': self.X})


if __name__ == "__main__":
    # disney
    fn_prefix = r"disney/"
    fn_a = fn_prefix + r"edges.csv"
    fn_x = fn_prefix + r"X2.mat"
    sp_flag = 0
    R = 8
    B = 8
    alpha = 0.8
    gamma1 = 0.15
    gamma2 = 0.15
    num_iter = 700      # max iteration
    thre = 0.001
    count_exceedThre = 2
    check_iter = math.ceil(num_iter / 5)
    epsilon_start = 0.04
    epsilon_end = 0.0035

    fn_w = fn_x.split('.')[0] + "_" + r"W.mat"
    fn_h = fn_x.split('.')[0] + "_" + r"H.mat"
    fn_x_out = fn_x.split('.')[0] + "_" + r"Xbins.mat"

    # Block DSGD
    dsgd = DSGD_CNMF(R, B, sp_flag, alpha, gamma1, gamma2, thre, count_exceedThre)
    dsgd.ReadInput(fn_a, fn_x)
    dsgd.ComputeDimension()

    # initial info
    f = open(fn_x.split('.')[0] + "_" + "log.txt", 'w')
    f.close()

    # start iterations
    dsgd.BDSGD_t(num_iter, check_iter, epsilon_start, epsilon_end)

    dsgd.SaveWH(fn_w, fn_h)
    dsgd.SaveX(fn_x_out)

