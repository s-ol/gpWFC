import pyopencl as cl
import pyopencl.array
import pyopencl.cltypes
import numpy as np

# have:      0b1110101           0b1110101
# can:       0b0100010   forbid: 0b1011101
# can:       0b1000100   forbid: 0b0111011
# OR^^       0b1100110   AND^^   0b0011001
# need:      0b1100100           0b1100100

def xyt2i(i):
	return i[0] * 8 + i[1]

def xy2i(x, y):
	x = x % 8
	y = y % 8
	return y * 8 + x

def i2xy(i):
	x = i // 8
	return (i // 8, i % 8)

# left, up, right, down
class WFCPropagator(object):
	def get_neighbours(self, x, y, direction): # @TODO 3d
		if direction == 0:
			return xy2i(x - 1, y)
		elif direction == 1:
			return xy2i(x, y - 1)
		elif direction == 2:
			return xy2i(x + 1, y)
		else:
			return xy2i(x, y + 1)

	def get_neighbourst(self, i, direction):
		x, y = i
		if direction == 0:
			return ((x - 1) % 8, y)
		elif direction == 1:
			return (x, (y - 1) % 8)
		elif direction == 2:
			return ((x + 1) % 8, y)
		else:
			return (x, (y + 1) % 8)

	def get_allows(self, i, direction):
		ret = 0
		tile = self.model.tiles[i]
		for other in self.model.tiles:
			if tile.compatible(other, direction):
				ret |= other.flag
		return ret

	def __init__(self, ctx, model):
		self.ctx = ctx
		self.model = model
		self.setup2d()

		self.neighbours = np.fromfunction(np.vectorize(self.get_neighbours), self.model.world_shape + (4,), dtype=int) # cl.cltypes.uint)
		self.allows = np.fromfunction(np.vectorize(self.get_allows), (len(self.model.tiles), 4), dtype=int) # cl.cltypes.uint)

		return
		self.allows_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.allows)
		self.neighbours_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.neighbours)

		self.program = cl.Program(ctx, self.preamble + '''
		__kernel void reduce_to_allowed(
			const uint i, const uint allowmap,
			__global uint* grid, __constant ADJUINT* allows, __constant ADJUINT* neighbours
		) {
			uint old_bits = grid[i];
			uint new_bits = old_bits & allowmap;
			grid[i] = new_bits;
			uint diff = old_bits ^ new_bits;
			if (!diff) return;

			ADJUINT allowmaps;
			for (int bit = 0; bit < STATES; bit++) {
				if (diff & (1 << i))
					allowmaps |= allows[bit];
			}

			// change in bit, trigger neighbours
			enqueue_kernel(
				get_default_queue(),
				CLK_ENQUEUE_FLAGS_WAIT_KERNEL,
				ndrange_1D(ADJ),
				^{
					uint neighbour = get_global_id(0);
					uint allow = allowmaps[neighbour];
					reduce_to_allowed(
						neighbours[i][neighbour], allow,
						grid, allows, neighbours
					);
				}
			);
		}
		''').build()

	def reduce_to_allowed(self, i, allowmap, grid):
		old = grid[i]
		new = old & allowmap
		diff = old ^ new
		# print('tile {}: {} & {} = {}, delta: {}'.format(i, old, allowmap, new, diff))
		grid[i] = new
		if not diff or not new:
			return

		allowmaps = np.zeros((4,), dtype=int)
		for tile in self.model.tiles:
			if new & tile.flag:
				# print('delta bit {}, propagate allows {}'.format(tile.index, self.allows[tile.index]))
				allowmaps |= self.allows[tile.index]
		# print('neighbour allows: {}'.format(allowmaps))

		for neighbour in range(4):
			self.reduce_to_allowed(
				# self.neighbours[i][neighbour], allowmaps[neighbour],
				self.get_neighbourst(i, neighbour), allowmaps[neighbour],
				grid
			)

	def setup2d(self):
		self.preamble = '''
			#define ADJ 4
			#define ADJUINT uint4
			#define STATES {}
		'''.format(len(self.model.tiles))

	def setup3d(self):
		self.preamble = '''
			#define ADJ 6
			#define ADJUINT uint8
			#define STATES {}
		'''.format(len(self.model.tiles))

	def propagate(self, grid, index, collapsed):
		# print('prop', index, collapsed, grid[index])
		self.reduce_to_allowed(index, collapsed, grid)

		return
		with cl.CommandQueue(self.ctx) as queue:
			self.program.reduce_to_allowed(
				queue, (1,), None,
				xyt2i(index), collapsed,
				grid, self.allows_buf, self.neighbours_buf
			)
