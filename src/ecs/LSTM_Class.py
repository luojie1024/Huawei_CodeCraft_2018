# -*- coding: utf-8 -*-
import math
import random
import vectorization as vt


class LSTM:
    def __init__(self, input_size, output_size, batch_size):
        # input size
        self.input_size = input_size
        # output size
        self.output_size = output_size
        # time steps
        self.batch_size = batch_size
        # gate = sigmoid( W*x<t> + U*h<t-1> + b )
        # parameters of forget gate pading zeros
        self.Wf = [[0] * (self.input_size) for i in range(self.output_size)]  # 11X9
        self.Uf = [[0] * (self.output_size) for i in range(self.output_size)]  # 9x9
        self.bf = [0] * self.output_size
        # parameters of input gate
        self.Wi = [[0] * (self.input_size) for i in range(self.output_size)]
        self.Ui = [[0] * (self.output_size) for i in range(self.output_size)]
        self.bi = [0] * self.output_size
        # parameters of output gate
        self.Wo = [[0] * (self.input_size) for i in range(self.output_size)]
        self.Uo = [[0] * (self.output_size) for i in range(self.output_size)]
        self.bo = [0] * self.output_size
        # parameters of new candidate cell
        self.Wc = [[0] * (self.input_size) for i in range(self.output_size)]
        self.Uc = [[0] * (self.output_size) for i in range(self.output_size)]
        self.bc = [0] * self.output_size
        # all attributes below are stored in the form of row vector
        # i.e. each row represents one time step within one batch
        # memory cells
        self.c = [[0] * self.output_size for i in range(self.batch_size + 1)]
        # hidden states
        self.h = [[0] * self.output_size for i in range(self.batch_size + 1)]
        # MDN parameters
        self.y = [[0] * self.output_size for i in range(self.batch_size + 1)]
        # gate and candidate c_tilde
        self.f_gate = [[0] * self.output_size for i in range(self.batch_size + 1)]
        self.i_gate = [[0] * self.output_size for i in range(self.batch_size + 1)]
        self.o_gate = [[0] * self.output_size for i in range(self.batch_size + 1)]
        self.c_tilde = [[0] * self.output_size for i in range(self.batch_size + 1)]
        # derivative of gate and c c_tilde and x
        self.delta_f = [[0] * self.output_size for i in range(self.batch_size + 1)]
        self.delta_i = [[0] * self.output_size for i in range(self.batch_size + 1)]
        self.delta_o = [[0] * self.output_size for i in range(self.batch_size + 1)]
        self.delta_c_tilde = [[0] * self.output_size for i in range(self.batch_size + 1)]
        self.delta_c = [[0] * self.output_size for i in range(self.batch_size + 1)]
        self.delta_x = [[0] * self.input_size for i in range(self.batch_size + 1)]
        # attributes for Adam
        self.counter = 0
        self.m_Wc = [[0] * (self.input_size) for i in range(self.output_size)]
        self.m_Wi = [[0] * (self.input_size) for i in range(self.output_size)]
        self.m_Wf = [[0] * (self.input_size) for i in range(self.output_size)]
        self.m_Wo = [[0] * (self.input_size) for i in range(self.output_size)]

        self.m_Uc = [[0] * (self.output_size) for i in range(self.output_size)]
        self.m_Ui = [[0] * (self.output_size) for i in range(self.output_size)]
        self.m_Uf = [[0] * (self.output_size) for i in range(self.output_size)]
        self.m_Uo = [[0] * (self.output_size) for i in range(self.output_size)]

        self.m_bc = [0] * self.output_size
        self.m_bi = [0] * self.output_size
        self.m_bf = [0] * self.output_size
        self.m_bo = [0] * self.output_size

        self.v_Wc = [[0] * (self.input_size) for i in range(self.output_size)]
        self.v_Wi = [[0] * (self.input_size) for i in range(self.output_size)]
        self.v_Wf = [[0] * (self.input_size) for i in range(self.output_size)]
        self.v_Wo = [[0] * (self.input_size) for i in range(self.output_size)]

        self.v_Uc = [[0] * (self.output_size) for i in range(self.output_size)]
        self.v_Ui = [[0] * (self.output_size) for i in range(self.output_size)]
        self.v_Uf = [[0] * (self.output_size) for i in range(self.output_size)]
        self.v_Uo = [[0] * (self.output_size) for i in range(self.output_size)]

        self.v_bc = [0] * self.output_size
        self.v_bi = [0] * self.output_size
        self.v_bf = [0] * self.output_size
        self.v_bo = [0] * self.output_size

    # initialize LSTM
    def init_lstm(self, rand_type='uniform'):
        #        random.seed(2)
        if rand_type == 'uniform':
            scale_w = math.sqrt(3.0 / self.input_size)
            scale_u = math.sqrt(3.0 / self.output_size)
            for i in range(len(self.Wf)):
                for j in range(len(self.Wf[0])):
                    self.Wf[i][j] = random.uniform(-scale_w, scale_w)
                    self.Wi[i][j] = random.uniform(-scale_w, scale_w)
                    self.Wo[i][j] = random.uniform(-scale_w, scale_w)
                    self.Wc[i][j] = random.uniform(-scale_w, scale_w)
                for j in range(len(self.Uf[0])):
                    self.Uf[i][j] = random.uniform(-scale_u, scale_u)
                    self.Ui[i][j] = random.uniform(-scale_u, scale_u)
                    self.Uo[i][j] = random.uniform(-scale_u, scale_u)
                    self.Uc[i][j] = random.uniform(-scale_u, scale_u)
                self.bf[i] = 1
                self.bi[i] = random.random()
                self.bo[i] = random.random()
                self.bc[i] = random.random()
        elif rand_type == 'gauss':
            std_w = math.sqrt(float(self.input_size))
            std_u = math.sqrt(float(self.output_size))
            for i in range(len(self.Wf)):
                for j in range(len(self.Wf[0])):
                    self.Wf[i][j] = random.gauss(0, std_w)
                    self.Wi[i][j] = random.gauss(0, std_w)
                    self.Wo[i][j] = random.gauss(0, std_w)
                    self.Wc[i][j] = random.gauss(0, std_w)
                for j in range(len(self.Uf[0])):
                    self.Uf[i][j] = random.gauss(0, std_u)
                    self.Ui[i][j] = random.gauss(0, std_u)
                    self.Uo[i][j] = random.gauss(0, std_u)
                    self.Uc[i][j] = random.gauss(0, std_u)
                self.bf[i] = 1
                self.bi[i] = random.random()
                self.bo[i] = random.random()
                self.bc[i] = random.random()
        else:
            raise Exception("Invalid initialization")

    # set new batch size
    def set_batch_size(self, new_batch_size):
        self.batch_size = new_batch_size
        # update dimensions of the following attributes
        # memory cells
        self.c = [[0] * self.output_size for i in range(self.batch_size + 1)]
        # hidden states
        self.h = [[0] * self.output_size for i in range(self.batch_size + 1)]
        # MDN parameters
        self.y = [[0] * self.output_size for i in range(self.batch_size + 1)]
        # gate and candidate c_tilde
        self.f_gate = [[0] * self.output_size for i in range(self.batch_size + 1)]
        self.i_gate = [[0] * self.output_size for i in range(self.batch_size + 1)]
        self.o_gate = [[0] * self.output_size for i in range(self.batch_size + 1)]
        self.c_tilde = [[0] * self.output_size for i in range(self.batch_size + 1)]
        # derivative of gate and c c_tilde and x
        self.delta_f = [[0] * self.output_size for i in range(self.batch_size + 1)]
        self.delta_i = [[0] * self.output_size for i in range(self.batch_size + 1)]
        self.delta_o = [[0] * self.output_size for i in range(self.batch_size + 1)]
        self.delta_c_tilde = [[0] * self.output_size for i in range(self.batch_size + 1)]
        self.delta_c = [[0] * self.output_size for i in range(self.batch_size + 1)]
        self.delta_x = [[0] * self.input_size for i in range(self.batch_size + 1)]

    # get memory cell
    def get_c(self):
        return self.c

    # get hidden units
    def get_h(self):
        return self.h

    # get MDN parameters
    def get_MDN_para(self):
        return self.y

    # get gates
    def get_gates(self):
        gates = {}
        gates['f'] = self.f_gate
        gates['i'] = self.i_gate
        gates['o'] = self.o_gate
        gates['c_tilde'] = self.c_tilde
        return gates

    # get derivatives
    def get_delta_gates(self):
        delta = {}
        delta['f'] = self.delta_f
        delta['i'] = self.delta_i
        delta['o'] = self.delta_o
        delta['c'] = self.delta_c
        delta['c_tilde'] = self.delta_c_tilde
        return delta

    # get parameters
    def get_parameters(self):
        parameters = {}
        parameters['Wf'] = self.Wf
        parameters['Uf'] = self.Uf
        parameters['bf'] = self.bf
        parameters['Wi'] = self.Wi
        parameters['Ui'] = self.Ui
        parameters['bi'] = self.bi
        parameters['Wo'] = self.Wo
        parameters['Uo'] = self.Uo
        parameters['bo'] = self.bo
        parameters['Wc'] = self.Wc
        parameters['Uc'] = self.Uc
        parameters['bc'] = self.bc
        return parameters

    # get delta_x for back propagation through layers
    def get_delta_x(self):
        return self.delta_x

    # forward propagation for one single LSTM cell instant t    
    def lstm_cell_forward(self, Xt, t):
        '''
        :param Xt: t时刻输入
        :param t: t时刻
        '''
        # get cell and hidden states in previous time step
        c_prev = self.c[t - 1]
        h_prev = self.h[t - 1]
        # forget gate
        forget_gate = vt.sigmoid(vt.add(vt.dot(self.Wf, Xt), \
                                        vt.add(vt.dot(self.Uf, h_prev), self.bf)))
        # input gate
        input_gate = vt.sigmoid(vt.add(vt.dot(self.Wi, Xt), \
                                       vt.add(vt.dot(self.Ui, h_prev), self.bi)))
        # output gate
        output_gate = vt.sigmoid(vt.add(vt.dot(self.Wo, Xt), \
                                        vt.add(vt.dot(self.Uo, h_prev), self.bo)))
        # c_tilde
        c_tilde = vt.tanh(vt.add(vt.dot(self.Wc, Xt), \
                                 vt.add(vt.dot(self.Uc, h_prev), self.bc)))
        # update c<t> and h<t>
        c_t = vt.add(vt.times(forget_gate, c_prev), vt.times(input_gate, c_tilde))
        h_t = vt.times(output_gate, vt.tanh(c_t))
        # update all relevant attributes
        self.f_gate[t] = forget_gate
        self.i_gate[t] = input_gate
        self.o_gate[t] = output_gate
        self.c_tilde[t] = c_tilde
        self.c[t] = c_t
        self.h[t] = h_t

    # forward propagation over the entire LSTM
    def lstm_forward(self, X):
        '''
        :param X: X数据块
        :return: 遍历单个数据,前向传播
        '''
        for t in range(1, self.batch_size + 1):
            self.lstm_cell_forward(X[t], t)

    # reset hidden states and cell memories after training over one mini-batch
    def reset(self):
        '''
        des:置为0
        '''
        for i in range(len(self.c)):
            for j in range(len(self.c[0])):
                self.c[i][j] = 0
                self.h[i][j] = 0

    # backward propagation for one single LSTM cell from t to t-1
    # Back Propagation Through Time
    def lstm_cell_bptt(self, Dt, D_out_t, t):
        delta_out_t = vt.add(Dt, D_out_t)
        # if from the last step back pass, then delta_c_<t+1> = 0, delta_f_<t+1> = 0
        if t == self.batch_size:
            delta_c_t = vt.times(delta_out_t, vt.times(self.o_gate[t], \
                                                       vt.sub(1, vt.times(vt.tanh(self.c[t]), \
                                                                          vt.tanh(self.c[t])))))
        else:
            delta_c_t = vt.add(vt.times(delta_out_t, vt.times(self.o_gate[t], \
                                                              vt.sub(1, vt.times(vt.tanh(self.c[t]), \
                                                                                 vt.tanh(self.c[t]))))), \
                               vt.times(self.delta_c[t + 1], self.f_gate[t + 1]))

        delta_c_tilde_t = vt.times(delta_c_t, vt.times(self.i_gate[t], \
                                                       vt.sub(1, vt.times(self.c_tilde[t], self.c_tilde[t]))))
        delta_i_t = vt.times(delta_c_t, vt.times(self.c_tilde[t], \
                                                 vt.times(self.i_gate[t], vt.sub(1, self.i_gate[t]))))
        delta_f_t = vt.times(delta_c_t, vt.times(self.c[t - 1], \
                                                 vt.times(self.f_gate[t], vt.sub(1, self.f_gate[t]))))
        delta_o_t = vt.times(delta_out_t, vt.times(vt.tanh(self.c[t]), \
                                                   vt.times(self.o_gate[t], vt.sub(1, self.o_gate[t]))))
        # update attributes
        self.delta_c[t] = delta_c_t
        self.delta_c_tilde[t] = delta_c_tilde_t
        self.delta_i[t] = delta_i_t
        self.delta_f[t] = delta_f_t
        self.delta_o[t] = delta_o_t
        # calculate D_out for next back pass and delta_x to pass to former layers
        U = []
        U.extend(self.Uc)
        U.extend(self.Ui)
        U.extend(self.Uf)
        U.extend(self.Uo)
        W = []
        W.extend(self.Wc)
        W.extend(self.Wi)
        W.extend(self.Wf)
        W.extend(self.Wo)
        delta_gate = []
        delta_gate.extend(delta_c_tilde_t)
        delta_gate.extend(delta_i_t)
        delta_gate.extend(delta_f_t)
        delta_gate.extend(delta_o_t)
        self.delta_x[t] = vt.dot(vt.transpose(W), delta_gate)
        D_out_t_minus_1 = vt.dot(vt.transpose(U), delta_gate)
        return D_out_t_minus_1

    # backward propagation over the entire LSTM and do gradient descent
    def lstm_bptt(self, D, X, learning_rate, optimizer='gradient_descent', \
                  beta1=0.9, beta2=0.999, epsilon=1e-8):
        # BPTT
        D_out_t = [0] * self.output_size
        for t in range(self.batch_size, 0, -1):
            D_out_t = self.lstm_cell_bptt(D[t], D_out_t, t)
        # calculate gradients
        delta_W = [[0] * (self.input_size) for i in range(4 * self.output_size)]
        delta_U = [[0] * (self.output_size) for i in range(4 * self.output_size)]
        delta_b = [0] * 4 * self.output_size
        for t in range(1, self.batch_size + 1):
            delta_gate = []
            delta_gate.extend(self.delta_c_tilde[t])
            delta_gate.extend(self.delta_i[t])
            delta_gate.extend(self.delta_f[t])
            delta_gate.extend(self.delta_o[t])
            delta_W = vt.add(delta_W, vt.cross(delta_gate, X[t]))
            delta_U = vt.add(delta_U, vt.cross(delta_gate, self.h[t - 1]))
            delta_b = vt.add(delta_b, delta_gate)
        # get gradients for each parameters
        # weight matrix
        d_Wc = delta_W[0:self.output_size]  # delta_W[Wc,Wi,Wf,Wo]
        d_Wi = delta_W[self.output_size:2 * self.output_size]
        d_Wf = delta_W[2 * self.output_size:3 * self.output_size]
        d_Wo = delta_W[3 * self.output_size:4 * self.output_size]
        # update matrix
        d_Uc = delta_U[0:self.output_size]  # delta_U[Uc,Ui,Uf,Uo]
        d_Ui = delta_U[self.output_size:2 * self.output_size]
        d_Uf = delta_U[2 * self.output_size:3 * self.output_size]
        d_Uo = delta_U[3 * self.output_size:4 * self.output_size]
        # bias
        d_bc = delta_b[0:self.output_size]  # delta_b[bc,bi,bf,bo]
        d_bi = delta_b[self.output_size:2 * self.output_size]
        d_bf = delta_b[2 * self.output_size:3 * self.output_size]
        d_bo = delta_b[3 * self.output_size:4 * self.output_size]
        if optimizer == 'gradient_descent':
            # gradient descent
            self.Wc = vt.sub(self.Wc, vt.times(learning_rate, d_Wc))
            self.Wi = vt.sub(self.Wi, vt.times(learning_rate, d_Wi))
            self.Wf = vt.sub(self.Wf, vt.times(learning_rate, d_Wf))
            self.Wo = vt.sub(self.Wo, vt.times(learning_rate, d_Wo))
            self.Uc = vt.sub(self.Uc, vt.times(learning_rate, d_Uc))
            self.Ui = vt.sub(self.Ui, vt.times(learning_rate, d_Ui))
            self.Uf = vt.sub(self.Uf, vt.times(learning_rate, d_Uf))
            self.Uo = vt.sub(self.Uo, vt.times(learning_rate, d_Uo))
            self.bc = vt.sub(self.bc, vt.times(learning_rate, d_bc))
            self.bi = vt.sub(self.bi, vt.times(learning_rate, d_bi))
            self.bf = vt.sub(self.bf, vt.times(learning_rate, d_bf))
            self.bo = vt.sub(self.bo, vt.times(learning_rate, d_bo))
        elif optimizer == 'Adam':
            self.counter = self.counter + 1

            self.m_Wc = vt.add(vt.times(beta1, self.m_Wc), vt.times(1 - beta2, d_Wc))
            self.m_Wi = vt.add(vt.times(beta1, self.m_Wi), vt.times(1 - beta2, d_Wi))
            self.m_Wf = vt.add(vt.times(beta1, self.m_Wf), vt.times(1 - beta2, d_Wf))
            self.m_Wo = vt.add(vt.times(beta1, self.m_Wo), vt.times(1 - beta2, d_Wo))

            self.m_Uc = vt.add(vt.times(beta1, self.m_Uc), vt.times(1 - beta2, d_Uc))
            self.m_Ui = vt.add(vt.times(beta1, self.m_Ui), vt.times(1 - beta2, d_Ui))
            self.m_Uf = vt.add(vt.times(beta1, self.m_Uf), vt.times(1 - beta2, d_Uf))
            self.m_Uo = vt.add(vt.times(beta1, self.m_Uo), vt.times(1 - beta2, d_Uo))

            self.m_bc = vt.add(vt.times(beta1, self.m_bc), vt.times(1 - beta2, d_bc))
            self.m_bi = vt.add(vt.times(beta1, self.m_bi), vt.times(1 - beta2, d_bi))
            self.m_bf = vt.add(vt.times(beta1, self.m_bf), vt.times(1 - beta2, d_bf))
            self.m_bo = vt.add(vt.times(beta1, self.m_bo), vt.times(1 - beta2, d_bo))

            self.v_Wc = vt.add(vt.times(beta1, self.v_Wc), vt.times(1 - beta2, vt.times(d_Wc, d_Wc)))
            self.v_Wi = vt.add(vt.times(beta1, self.v_Wi), vt.times(1 - beta2, vt.times(d_Wi, d_Wi)))
            self.v_Wf = vt.add(vt.times(beta1, self.v_Wf), vt.times(1 - beta2, vt.times(d_Wf, d_Wf)))
            self.v_Wo = vt.add(vt.times(beta1, self.v_Wo), vt.times(1 - beta2, vt.times(d_Wo, d_Wo)))

            self.v_Uc = vt.add(vt.times(beta1, self.v_Uc), vt.times(1 - beta2, vt.times(d_Uc, d_Uc)))
            self.v_Ui = vt.add(vt.times(beta1, self.v_Ui), vt.times(1 - beta2, vt.times(d_Ui, d_Ui)))
            self.v_Uf = vt.add(vt.times(beta1, self.v_Uf), vt.times(1 - beta2, vt.times(d_Uf, d_Uf)))
            self.v_Uo = vt.add(vt.times(beta1, self.v_Uo), vt.times(1 - beta2, vt.times(d_Uo, d_Uo)))

            self.v_bc = vt.add(vt.times(beta1, self.v_bc), vt.times(1 - beta2, vt.times(d_bc, d_bc)))
            self.v_bi = vt.add(vt.times(beta1, self.v_bi), vt.times(1 - beta2, vt.times(d_bi, d_bi)))
            self.v_bf = vt.add(vt.times(beta1, self.v_bf), vt.times(1 - beta2, vt.times(d_bf, d_bf)))
            self.v_bo = vt.add(vt.times(beta1, self.v_bo), vt.times(1 - beta2, vt.times(d_bo, d_bo)))

            m_Wc_correct = vt.times(1 / (1 - math.pow(beta1, self.counter)), self.m_Wc)
            m_Wi_correct = vt.times(1 / (1 - math.pow(beta1, self.counter)), self.m_Wi)
            m_Wf_correct = vt.times(1 / (1 - math.pow(beta1, self.counter)), self.m_Wf)
            m_Wo_correct = vt.times(1 / (1 - math.pow(beta1, self.counter)), self.m_Wo)

            m_Uc_correct = vt.times(1 / (1 - math.pow(beta1, self.counter)), self.m_Uc)
            m_Ui_correct = vt.times(1 / (1 - math.pow(beta1, self.counter)), self.m_Ui)
            m_Uf_correct = vt.times(1 / (1 - math.pow(beta1, self.counter)), self.m_Uf)
            m_Uo_correct = vt.times(1 / (1 - math.pow(beta1, self.counter)), self.m_Uo)

            m_bc_correct = vt.times(1 / (1 - math.pow(beta1, self.counter)), self.m_bc)
            m_bi_correct = vt.times(1 / (1 - math.pow(beta1, self.counter)), self.m_bi)
            m_bf_correct = vt.times(1 / (1 - math.pow(beta1, self.counter)), self.m_bf)
            m_bo_correct = vt.times(1 / (1 - math.pow(beta1, self.counter)), self.m_bo)

            v_Wc_correct = vt.times(1 / (1 - math.pow(beta2, self.counter)), self.v_Wc)
            v_Wi_correct = vt.times(1 / (1 - math.pow(beta2, self.counter)), self.v_Wi)
            v_Wf_correct = vt.times(1 / (1 - math.pow(beta2, self.counter)), self.v_Wf)
            v_Wo_correct = vt.times(1 / (1 - math.pow(beta2, self.counter)), self.v_Wo)

            v_Uc_correct = vt.times(1 / (1 - math.pow(beta2, self.counter)), self.v_Uc)
            v_Ui_correct = vt.times(1 / (1 - math.pow(beta2, self.counter)), self.v_Ui)
            v_Uf_correct = vt.times(1 / (1 - math.pow(beta2, self.counter)), self.v_Uf)
            v_Uo_correct = vt.times(1 / (1 - math.pow(beta2, self.counter)), self.v_Uo)

            v_bc_correct = vt.times(1 / (1 - math.pow(beta2, self.counter)), self.v_bc)
            v_bi_correct = vt.times(1 / (1 - math.pow(beta2, self.counter)), self.v_bi)
            v_bf_correct = vt.times(1 / (1 - math.pow(beta2, self.counter)), self.v_bf)
            v_bo_correct = vt.times(1 / (1 - math.pow(beta2, self.counter)), self.v_bo)

            self.Wc = vt.sub(self.Wc, vt.times(learning_rate, vt.times(m_Wc_correct, vt.inverse(
                vt.add(vt.sq_root(v_Wc_correct), epsilon)))))
            self.Wi = vt.sub(self.Wi, vt.times(learning_rate, vt.times(m_Wi_correct, vt.inverse(
                vt.add(vt.sq_root(v_Wi_correct), epsilon)))))
            self.Wf = vt.sub(self.Wf, vt.times(learning_rate, vt.times(m_Wf_correct, vt.inverse(
                vt.add(vt.sq_root(v_Wf_correct), epsilon)))))
            self.Wo = vt.sub(self.Wo, vt.times(learning_rate, vt.times(m_Wo_correct, vt.inverse(
                vt.add(vt.sq_root(v_Wo_correct), epsilon)))))

            self.Uc = vt.sub(self.Uc, vt.times(learning_rate, vt.times(m_Uc_correct, vt.inverse(
                vt.add(vt.sq_root(v_Uc_correct), epsilon)))))
            self.Ui = vt.sub(self.Ui, vt.times(learning_rate, vt.times(m_Ui_correct, vt.inverse(
                vt.add(vt.sq_root(v_Ui_correct), epsilon)))))
            self.Uf = vt.sub(self.Uf, vt.times(learning_rate, vt.times(m_Uf_correct, vt.inverse(
                vt.add(vt.sq_root(v_Uf_correct), epsilon)))))
            self.Uo = vt.sub(self.Uo, vt.times(learning_rate, vt.times(m_Uo_correct, vt.inverse(
                vt.add(vt.sq_root(v_Uo_correct), epsilon)))))

            self.bc = vt.sub(self.bc, vt.times(learning_rate, vt.times(m_bc_correct, vt.inverse(
                vt.add(vt.sq_root(v_bc_correct), epsilon)))))
            self.bi = vt.sub(self.bi, vt.times(learning_rate, vt.times(m_bi_correct, vt.inverse(
                vt.add(vt.sq_root(v_bi_correct), epsilon)))))
            self.bf = vt.sub(self.bf, vt.times(learning_rate, vt.times(m_bf_correct, vt.inverse(
                vt.add(vt.sq_root(v_bf_correct), epsilon)))))
            self.bo = vt.sub(self.bo, vt.times(learning_rate, vt.times(m_bo_correct, vt.inverse(
                vt.add(vt.sq_root(v_bo_correct), epsilon)))))

    # Least Square Error for step t
    def loss_LSE(self, label_t, t):
        return vt.norm2_sq(vt.sub(self.h[t], label_t)) / 2

    # Least Square Cost over one batch
    def cost_LSE(self, Label):
        cost = 0
        for t in range(1, self.batch_size + 1):
            cost = cost + self.loss_LSE(Label[t], t)
        return cost

    # Derivatives Direct Output
    def deriv_LSE(self, Label):
        deriv = [[0] * self.output_size for i in range(self.batch_size + 1)]
        # calculate derivatives with respect to each time step within one batch
        for t in range(1, self.batch_size + 1):
            deriv[t] = vt.sub(self.h[t], Label[t])
        return deriv

    # Prediction @ step t
    def predict_t(self, c_prev, h_prev, Xt):
        # forget gate
        forget_gate = vt.sigmoid(vt.add(vt.dot(self.Wf, Xt), \
                                        vt.add(vt.dot(self.Uf, h_prev), self.bf)))
        # input gate
        input_gate = vt.sigmoid(vt.add(vt.dot(self.Wi, Xt), \
                                       vt.add(vt.dot(self.Ui, h_prev), self.bi)))
        # output gate
        output_gate = vt.sigmoid(vt.add(vt.dot(self.Wo, Xt), \
                                        vt.add(vt.dot(self.Uo, h_prev), self.bo)))
        # c_tilde
        c_tilde = vt.tanh(vt.add(vt.dot(self.Wc, Xt), \
                                 vt.add(vt.dot(self.Uc, h_prev), self.bc)))
        # update c<t> and h<t>
        c_t = vt.add(vt.times(forget_gate, c_prev), vt.times(input_gate, c_tilde))
        h_t = vt.times(output_gate, vt.tanh(c_t))
        return c_t, h_t
