import pyopencl as cl
import pyopencl.array
import pyopencl.clrandom
import pyopencl.reduction
import numpy as np
import numpy.random

platform = None
platforms =  cl.get_platforms()
if len(platforms) == 0:
	platform = platforms[0]
for pl in platforms:
	if 'NVIDIA' in pl.name:
		platform = pl

device = platform.get_devices()[0]
ctx = cl.Context([ device ])
queue = cl.CommandQueue(ctx)
rnd = pyopencl.clrandom.PhiloxGenerator(ctx)

print('using {} CL platform'.format(platform.name))

world_shape = (8, 8,)

class Tile:
	nextIndex = 0
	def __init__(self, char, adj, weight=0.2):
		self.char = char
		self.adj = adj
		self.weight = weight
		self.index = Tile.nextIndex
		Tile.nextIndex += 1
		self.flag = 1 << self.index

tiles = [
	Tile('┃', [1, 1, 0, 0], 1),
	Tile('━', [0, 0, 1, 1]),
	Tile('┓', [0, 1, 1, 0]),
	Tile('┗', [1, 0, 0, 1]),
	Tile('┳', [0, 1, 1, 1]),
	Tile('┻', [1, 0, 1, 1]),
	Tile('╹', [1, 0, 0, 0]),
	Tile('╻', [0, 1, 0, 0]),
]
weights = cl.array.to_device(queue, np.array(list(tile.weight for tile in tiles), dtype=cl.cltypes.float))

bits = len(tiles)
if bits < 9:
	bitfield_dt = np.uint8
elif bits < 17:
	bitfield_dt = np.uint16
elif bits < 33:
	bitfield_dt = np.uint32
else:
	bitfield_dt = np.uint64
bitfield_dt = cl.cltypes.uint

print('using {} dtype ({} tiles)'.format(np.dtype(bitfield_dt).name, len(tiles)))
all_tiles = sum(tile.flag for tile in tiles)
print('filling grid with {}'.format(all_tiles))

grid = cl.array.to_device(queue, np.full(world_shape, all_tiles, dtype=bitfield_dt))
bias = cl.array.to_device(queue, np.zeros(world_shape, dtype=cl.cltypes.float))

def make_dtype():
	dtype = np.dtype([
		('entropy', cl.cltypes.float),
		('index', bitfield_dt),
	])

	name = 'min_collector'
	from pyopencl.tools import get_or_register_dtype, match_dtype_to_c_struct

	dtype, c_decl = match_dtype_to_c_struct(device, name, dtype)
	dtype = get_or_register_dtype(name, dtype)

	return dtype, c_decl

col_dt, col_cd = make_dtype()

find_lowest_entropy = cl.reduction.ReductionKernel(ctx,
	arguments='__global uint* grid, __global float* bias, const unsigned int states, __global float* weights',
	neutral='neutral()',
	dtype_out=col_dt,
	map_expr='get_entropy(i, grid[i], bias[i], states, weights)',
	reduce_expr='reduce(a, b)',
	preamble=col_cd + r'''//CL//

	/* start with an imaginary solved tile */
	min_collector neutral() {
		min_collector res;
		res.entropy = -1.0;
		res.index = 0;
		return res;
	}

	/* get entropy of tile (> 0)
	 * -1: solved
	 *  0: overconstrained */
	min_collector get_entropy(int i, uint bitfield, float bias, const unsigned int states, __global float* weights) {
		min_collector res;
		res.entropy = 0.0f;
		res.index = i;

		unsigned int remaining_states = 0;
		for (unsigned int state = 0; state < states; state++) {
			if (bitfield & (1 << state)) {
				remaining_states++;
				res.entropy += weights[state];
			}
		}

		if (remaining_states == 1) res.entropy = -1.0;
		else if (remaining_states > 1) res.entropy += bias * 0.5;
		return res;
	}

	/* if one is solved try the other
	 * otherwise reduce to minimum entropy */
	min_collector reduce(min_collector a, min_collector b) {
		if (a.entropy < 0.0) return b;
		if (b.entropy < 0.0) return a;
		return b.entropy < a.entropy ? b : a;
	}
	'''
)

def i2xy(i):
	x = i // 8
	return (x, i % 8,)

def collapse(index):
	bits = grid[index]
	p = numpy.array([tile.weight * (not not bits & tile.flag) for tile in tiles])
	p = p / np.sum(p)
	tile = np.random.choice(tiles, p=p)
	print('collapsing from {} to {}'.format(index, bits, tile.flag))
	grid[index] = tile.flag

def observe():
	# random tie-breaking bias for each tile
	rnd.fill_uniform(bias)

	tile = find_lowest_entropy(grid, bias, len(tiles), weights).get()
	entropy, index = tile['entropy'], i2xy(tile['index'])

	if entropy < 0:
		print('solved!'.format(index))
		return 'done'
	elif entropy == 0:
		print('tile {} overconstrained!'.format(index))
		return 'overconstrained'

	print('selected tile {} with entropy {}'.format(index, entropy))
	collapse(index)

	return 'continue'

observe()

vector = np.zeros((1, 1), cl.array.vec.float4)
matrix = np.zeros((1, 4), cl.array.vec.float4)
matrix[0, 0] = (1, 2, 4, 8)
matrix[0, 1] = (16, 32, 64, 128)
matrix[0, 2] = (3, 6, 9, 12)
matrix[0, 3] = (5, 10, 15, 25)
vector[0, 0] = (1, 2, 4, 8)
result = np.zeros(4, np.float32)

matrix_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=matrix)
vector_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=vector)
destination_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, result.nbytes)
