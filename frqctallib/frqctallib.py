import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import matplotlib as mpl
from matplotlib import cm, pyplot as plt
from PIL import Image

class Frqctal:

    def __init__(self, max_iter=100,
                 nrows=512, ncols=512,
                _nThreads=16):

        # various parameters
        self._max_iter = max_iter
        self._nrows = nrows
        self._ncols = ncols
        self._nThreads = _nThreads
        self._nBlocks = int(np.ceil(self._nrows/float(self._nThreads)))

        #our gpu function
        self._func_code = """
            __global__ void cump_fractal(int max_iter, float *xs, float *ys, int *nb_iters)
            {
                int idx = threadIdx.x + blockDim.x * blockIdx.x;

                int nb_iter = 0;
                float x0 = xs[idx];
                float y0 = ys[idx];
                float xtemp, x = 0, y = 0;

                if(idx >= %(NB_PIXELS)s)
                    return;

                while(x*x + y*y < 4 && nb_iter < max_iter)
                {
                    xtemp = x*x - y*y + x0;
                    y = 2*x*y + y0;
                    x = xtemp;
                    ++nb_iter;
                }

                nb_iters[idx] = nb_iter;
          }
        """ % {'NB_PIXELS':self._ncols*self._nrows}

        mod = SourceModule(self._func_code)
        self._gpu_func = mod.get_function("cump_fractal")

    def get_iterations(self, zoom = 1, center_x=0, center_y=0):

        max_x = 2**-zoom + center_x
        min_x = -2**-zoom + center_x
        max_y = 2**-zoom - center_y
        min_y = -2**-zoom - center_y

        xs = np.linspace(min_x, max_x, self._nrows, dtype=np.float32)
        ys = np.linspace(min_y, max_y, self._ncols, dtype=np.float32)

        xs = np.tile(xs, self._ncols) # 1, ..... 1, ....
        ys = np.repeat(ys, self._nrows) # 1, 1, ....

        iters = np.zeros_like(xs, np.int32)
        self._gpu_func(np.int32(self._max_iter), cuda.In(xs), cuda.In(ys), cuda.Out(iters),
                       block=(self._nThreads**2, 1, 1), grid=(self._nBlocks**2, 1))

        iters = iters.reshape(self._nrows, self._ncols)
        return self.get_img(iters)

    def get_img(self, iters):

        iters = 1 + np.log(iters / float(self._max_iter))
        im = Image.fromarray(cm.gist_earth(iters, bytes=True))
        return im


    def save_img(self, im, file_name = "frqctal.png"):
        im.save(file_name)
15
def main():

    fq = Frqctal(max_iter=400)
    print "gpu!!!"
    img = fq.get_iterations(4, -0.65013, 0.49185)
    fq.save_img(img)

main()