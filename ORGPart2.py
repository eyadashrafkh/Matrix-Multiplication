import numpy as np
import time
import threading
import matplotlib.pyplot as plt


class Matrix:
    def __init__(self, A, B, no_of_blocks):

        self.A = A
        self.B = B
        self.block_size = int(A.shape[1]/no_of_blocks)
        self.no_of_blocks = no_of_blocks
        self.blocks = np.empty((no_of_blocks, no_of_blocks), dtype=object)
        self.C = np.zeros_like(self.A)

    def release_threads(self):
        for i in range(self.no_of_blocks):
            for j in range(self.no_of_blocks):
                B1 = Block(i, j, self.block_size, self.A,
                           self.B, self.no_of_blocks)
                B1.thread.start()
                self.blocks[i][j] = B1
        B1.thread.join()
        for i in range(self.no_of_blocks):
            for j in range(self.no_of_blocks):
                self.C[i*self.block_size: i*self.block_size+self.block_size, j *
                       self.block_size: j*self.block_size+self.block_size] = self.blocks[i][j].block_C


class Block(object):
    def __init__(self, id_x, id_y, block_size, A, B, no_of_blocks):
        self.id_x = id_x
        self.id_y = id_y
        self.no_of_blocks = no_of_blocks

        self.start_row = id_x*block_size
        self.end_row = self.start_row+block_size

        self.start_column = id_y*block_size
        self.end_column = self.start_column+block_size

        self.block_size = block_size
        self.block_C = np.zeros_like((block_size, block_size))
        self.thread = threading.Thread(
            target=self.Calculate_block_multiplication, args=(A, B))

    def multiplyMatrices(self, A_block, B_block):
        r, c = A_block.shape
        r1, c1 = B_block.shape

        assert c == r1
        block_C = np.empty((r, c1))

        for i in range(r):
            for j in range(c):
                for k in range(c):
                    block_C[i][j] += A_block[i][k] * B_block[k][j]

        return np.matmul(A_block, B_block)

    def copy_block_from_matrix(self, AA, i_r, j_c):
        self.start_row = i_r*self.block_size
        self.end_row = self.start_row+self.block_size
        self.start_column = j_c*self.block_size
        self.end_column = self.start_column+self.block_size
        return AA[self.start_row:self.end_row, self.start_column:self.end_column]

    def Calculate_block_multiplication(self, A, B):
        print('Thread', self.id_x, self.id_y, ' released')
        for i in range(self.no_of_blocks):
            A1 = self.copy_block_from_matrix(A, self.id_x, i)
            B1 = self.copy_block_from_matrix(B, i, self.id_y)
            C1 = self.multiplyMatrices(A1, B1)
            self.block_C = np.add(self.block_C, C1)


np.random.seed = 32


def createMatrix(m, n):
    return np.random.randint(100, size=(m, n))


def multiplyMatrices(A, B):
    r, c = A.shape
    r1, c1 = B.shape
    C = np.empty((r, c1))
    np.matmul
    for i in range(r):
        for j in range(c):
            for k in range(c):
                C[i][j] += A[i][k] * B[k][j]
    return C

def block_matrix_multiplication(matrix_size,block_size):
    A = createMatrix(matrix_size, matrix_size)
    B = createMatrix(matrix_size, matrix_size)
    print(A)
    print()
    print(B)
    mm = Matrix(A, B, block_size)
    mm.release_threads()
    print()
    print()
    print(mm.blocks[0, 0].block_C)
    print()
    print(mm.C)

matrix_sizes = [2**n for n in range(2,8)]
time_blocks_pairs = dict()
for matrix_size in matrix_sizes:
    no_of_blocks = [min(2**n,matrix_size) for n in range(5)]
    timer = list()
    block_sizes = list()
    for block_size in no_of_blocks:
        t1 = time.perf_counter()
        thead1=threading.Thread(target=block_matrix_multiplication,args=(matrix_size,block_size))
        thead1.start()
        thead1.join()
        t2 = time.perf_counter()
        timer.append(t2-t1)
        block_sizes.append(block_size)
        time_blocks_pairs[matrix_size] = (block_sizes,timer)
for key in time_blocks_pairs:
    X = time_blocks_pairs[key][0]
    Y = time_blocks_pairs[key][1]
    plt.plot(X,Y,label=key)
plt.xlabel('Block Size')

plt.ylabel('Time (s)')
plt.title('Performance of Matrix Multiplication using blocking')
plt.legend()
plt.show()