import pyopencl as cl
import pyopencl.array
import pyopencl.clrandom
import pyopencl.tools
import pyopencl.reduction
import numpy as np
import numpy.random

def i2xy(i):
	x = i // 8
	return (i // 8, i % 8)

class WFCObserver(object):
	def __init__(self, ctx, queue, model):
		self.model = model
		self.rnd = pyopencl.clrandom.PhiloxGenerator(ctx)
		self.bias = cl.array.to_device(queue, np.zeros(self.model.world_shape, dtype=cl.cltypes.float))

		self.weights = cl.array.to_device(queue, np.array(list(tile.weight for tile in self.model.tiles), dtype=cl.cltypes.float))

		min_collector = np.dtype([
			('entropy', cl.cltypes.float),
			('index', cl.cltypes.uint),
		])
		min_collector, min_collector_def = cl.tools.match_dtype_to_c_struct(ctx.devices[0], 'min_collector', min_collector)
		min_collector = cl.tools.get_or_register_dtype('min_collector', min_collector)

		self.find_lowest_entropy = cl.reduction.ReductionKernel(ctx,
			arguments='__global uint* grid, __global float* bias, const uint states, __global float* weights',
			neutral='neutral()',
			dtype_out=min_collector,
			map_expr='get_entropy(i, grid[i], bias[i], states, weights)',
			reduce_expr='reduce(a, b)',
			preamble=min_collector_def + r'''//CL//

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

	def collapse(self, bits):
		p = self.weights.get()
		for i in range(len(p)):
			p[i] *= not not bits & (1 << i)
		p = p / np.sum(p)
		tile = np.random.choice(self.model.tiles, p=p)
		print('collapsing from {} to {}'.format(bits, tile.flag))
		return tile.flag

	def observe(self, grid):
		# random tie-breaking bias for each tile
		self.rnd.fill_uniform(self.bias)

		tile = self.find_lowest_entropy(grid, self.bias, len(self.weights), self.weights).get()
		entropy, index = tile['entropy'], i2xy(tile['index'])

		if entropy < 0:
			print('solved!'.format(index))
			return ('done',)
		elif entropy == 0:
			print('tile {} overconstrained!'.format(index))
			return ('error',)

		print('selected tile {} with entropy {}'.format(index, entropy))
		return ('continue', index, self.collapse(grid[index]))
