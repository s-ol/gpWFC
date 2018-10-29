import pyopencl as cl
import pyopencl.array
import pyopencl.clrandom
import pyopencl.tools
import pyopencl.reduction
import numpy as np
import numpy.random

class CLObserver(object):
	def __init__(self, model, ctx=None):
		self.model = model
		with cl.CommandQueue(ctx) as queue:
			self.rnd = pyopencl.clrandom.PhiloxGenerator(ctx)
			self.bias = cl.array.to_device(queue, np.zeros(self.model.world_shape, dtype=cl.cltypes.float))

			alloc = cl.tools.ImmediateAllocator(queue, mem_flags=cl.mem_flags.READ_ONLY)
			self.weights_array = np.array(list(tile.weight for tile in self.model.tiles), dtype=cl.cltypes.float)
			self.weights = cl.array.to_device(queue, self.weights_array, alloc)

		min_collector = np.dtype([
			('entropy', cl.cltypes.float),
			('index', cl.cltypes.uint),
		])
		min_collector, min_collector_def = cl.tools.match_dtype_to_c_struct(ctx.devices[0], 'min_collector', min_collector)
		min_collector = cl.tools.get_or_register_dtype('min_collector', min_collector)

		preamble = '''
			#define STATES {}
		'''.format(len(self.model.tiles))

		self.find_lowest_entropy = cl.reduction.ReductionKernel(ctx,
			arguments='__global ulong* grid, __global float* bias, __global float* weights',
			neutral='neutral()',
			dtype_out=min_collector,
			map_expr='get_entropy(i, grid[i], bias[i], weights)',
			reduce_expr='reduce(a, b)',
			preamble=min_collector_def + preamble + r'''//CL//

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
			min_collector get_entropy(uint i, ulong bitfield, float bias, __global float* weights) {
				min_collector res;
				res.entropy = 0.0f;
				res.index = i;

				uint remaining_states = 0;
				for (uint state = 0; state < STATES; state++) {
					if (bitfield & ((ulong)1 << state)) {
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
		p = self.weights_array.copy()
		bits = int(bits.get())
		for i in range(len(p)):
			p[i] *= not not bits & (1 << i)
		p = p / np.sum(p)
		tile = np.random.choice(self.model.tiles, p=p)
		print('collapsing from {} to {}'.format(bits, tile.flag))
		return tile.flag

	def observe(self, grid):
		# random tie-breaking bias for each tile
		self.rnd.fill_uniform(self.bias)

		tile = self.find_lowest_entropy(grid, self.bias, self.weights).get()
		entropy, index = tile['entropy'].item(), tile['index'].item()

		t_index = np.unravel_index(index, self.model.world_shape)
		if entropy < 0:
			print('solved!')
			return ('done',)
		elif entropy == 0:
			print('tile {} overconstrained!'.format(t_index))
			return ('error',)

		print('selected tile {} with entropy {}'.format(t_index, entropy))
		return ('continue', index, self.collapse(grid[t_index]))
