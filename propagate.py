import pyopencl as cl
import pyopencl.array
import pyopencl.cltypes
import numpy as np

def xyt2i(i):
	return i[1] * 8 + i[0]

def xy2i(x, y):
	x = x % 8
	y = y % 8
#	return y * 8 + x
	return x * 8 + y

def i2xy(i):
	x = i // 8
#	return (i % 8), i // 8
	return i //8, (i % 8)

class CPUPropagator(object):
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
		ret = np.uint64(0)
		tile = self.model.tiles[i]
		for other in self.model.tiles:
			if tile.compatible(other, direction):
				ret |= other.flag
		return ret

	def __init__(self, ctx, queue, model):
		self.ctx = ctx
		self.model = model

		self.neighbours = np.fromfunction(np.vectorize(self.get_neighbours), self.model.world_shape + (4,), dtype=int).astype(cl.cltypes.ulong)
		self.allows = np.fromfunction(np.vectorize(self.get_allows), (len(self.model.tiles), 4), dtype=int).astype(cl.cltypes.ulong)

	def reduce_to_allowed(self, i, allowmap, grid):
		old = grid[i]
		new = old & allowmap
		# print('tile {}: {} & {} = {}, delta: {}'.format(i, old, allowmap, new, diff))
		if old == new or not new:
			return
		grid[i] = new

		allowmaps = np.zeros((4,), dtype=np.uint64)
		for tile in self.model.tiles:
			if new & tile.flag:
				# print('delta bit {}, propagate allows {}'.format(tile.index, self.allows[tile.index]))
				allowmaps |= self.allows[tile.index]
		# print('neighbour allows: {}'.format(allowmaps))

		for neighbour in range(4):
			self.reduce_to_allowed(
				self.get_neighbourst(i, neighbour), allowmaps[neighbour],
				grid
			)

	def propagate(self, grid, index, collapsed):
		self.reduce_to_allowed(index, collapsed, grid)

class CL2Propagator(object):
	def get_neighbours(self, x, y, direction): # @TODO 3d
		if direction == 0:
			return xy2i(x - 1, y)
		elif direction == 1:
			return xy2i(x, y - 1)
		elif direction == 2:
			return xy2i(x + 1, y)
		else:
			return xy2i(x, y + 1)

	def get_allows(self, i, direction):
		ret = np.uint64(0)
		tile = self.model.tiles[i]
		for other in self.model.tiles:
			if tile.compatible(other, direction):
				ret |= other.flag
		return ret

	def __init__(self, ctx, model):
		self.ctx = ctx
		self.model = model
		self.setup2d()

		self.neighbours = np.fromfunction(np.vectorize(self.get_neighbours), self.model.world_shape + (4,), dtype=int).astype(cl.cltypes.ulong)
		self.allows = np.fromfunction(np.vectorize(self.get_allows), (len(self.model.tiles), 4), dtype=int).astype(cl.cltypes.ulong)

		self.allows_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.allows)
		self.neighbours_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.neighbours)

		self.program = cl.Program(ctx, self.preamble + '''
		__kernel void reduce_to_allowed(
			const uint i, const ulong allowmap,
			__global ulong* grid, __global ADJLONG* allows, __global ADJUINT* neighbours
		) {
			ulong old_bits = grid[i];
			ulong new_bits = old_bits & allowmap;
			grid[i] = new_bits;
			ulong diff = old_bits ^ new_bits;
			if (!diff) return;

			ADJLONG allowmaps;
			for (int bit = 0; bit < STATES; bit++) {
				if (new & (1 << i))
					allowmaps |= allows[bit];
			}

			// change in bit, trigger neighbours
			enqueue_kernel(
				get_default_queue(),
				CLK_ENQUEUE_FLAGS_WAIT_KERNEL,
				ndrange_1D(ADJ),
				^{
					uint neighbour = get_global_id(0);
					ulong allow = allowmaps[neighbour];
					reduce_to_allowed(
						neighbours[i][neighbour], allow,
						grid, allows, neighbours
					);
				}
			);
		}
		''').build()

	def setup2d(self):
		self.preamble = '''
			#define ADJ 4
			#define ADJUINT uint4
			#define ADJLONG ulong4
			#define STATES {}
		'''.format(len(self.model.tiles))

	def setup3d(self):
		self.preamble = '''
			#define ADJ 6
			#define ADJUINT uint8
			#define ADJLONG ulong8
			#define STATES {}
		'''.format(len(self.model.tiles))

	def propagate(self, grid, index, collapsed):
		with cl.CommandQueue(self.ctx) as queue:
			self.program.reduce_to_allowed(
				queue, (1,), None,
				xyt2i(index), collapsed,
				grid, self.allows_buf, self.neighbours_buf
			)

class CL1Propagator(object):
	def get_neighbours(self, x, y, direction): # @TODO 3d
		if direction == 0:
			return xy2i(x - 1, y)
		elif direction == 1:
			return xy2i(x, y - 1)
		elif direction == 2:
			return xy2i(x + 1, y)
		else:
			return xy2i(x, y + 1)

	def get_allows(self, i, direction):
		ret = np.uint64(0)
		tile = self.model.tiles[i]
		for other in self.model.tiles:
			if tile.compatible(other, direction):
				ret |= other.flag
		return ret

	def __init__(self, ctx, queue, model):
		self.ctx = ctx
		self.model = model
		self.setup2d()

		alloc = cl.tools.ImmediateAllocator(queue, mem_flags=cl.mem_flags.READ_ONLY)
		self.neighbours = np.fromfunction(np.vectorize(self.get_neighbours), self.model.world_shape + (4,), dtype=int).astype(cl.cltypes.uint)
		self.allows = np.fromfunction(np.vectorize(self.get_allows), (len(self.model.tiles), 4), dtype=int).astype(cl.cltypes.ulong)

		self.allows_buf = cl.array.to_device(queue, self.allows, alloc)
		self.neighbours_buf = cl.array.to_device(queue, self.neighbours, alloc)

		self.update_grid = cl.reduction.ReductionKernel(ctx,
			arguments='__global ulong* grid, __global ulong4* allows, __global uint4* neighbours',
			neutral='0',
			dtype_out=cl.cltypes.uint,
			map_expr='update_tile(i, grid, allows, neighbours)',
			reduce_expr='a + b',
			preamble=self.preamble + r'''//CL//
			uint update_tile(uint i, __global ulong* grid, __global ADJLONG* allows, __global ADJUINT* neighbours) {
				ulong old_bits = grid[i];

				ADJUINT next = neighbours[i];
				ulong left  = grid[next.x];
				ulong up    = grid[next.y];
				ulong right = grid[next.z];
				ulong down  = grid[next.w];
				ulong leftmask = 0;
				ulong upmask = 0;
				ulong rightmask = 0;
				ulong downmask = 0;

				for (uint tile = 0; tile < STATES; tile++) {
					ulong flag = 1 << tile;
					if (flag & right) rightmask |= allows[tile].x;
					if (flag &  down)  downmask |= allows[tile].y;
					if (flag &  left)  leftmask |= allows[tile].z;
					if (flag &    up)    upmask |= allows[tile].w;
				}

				ulong new_bits = old_bits & leftmask & upmask & rightmask & downmask;
				// new_bits = next.x; // old_bits & left;
				if (old_bits == new_bits) return 0;
				grid[i] = new_bits;
				return 1;
			}
                        '''
		)

	def setup2d(self):
		self.preamble = '''
			#define ADJ 4
			#define ADJUINT uint4
			#define ADJLONG ulong4
			#define STATES {}
		'''.format(len(self.model.tiles))

	def setup3d(self):
		self.preamble = '''
			#define ADJ 6
			#define ADJUINT uint8
			#define ADJLONG ulong8
			#define STATES {}
		'''.format(len(self.model.tiles))

	def propagate(self, grid, index, collapsed):
		grid[index] = collapsed
		turn, changes = 0, 1
		while changes > 0:
			changes = self.update_grid(grid, self.allows_buf, self.neighbours_buf).get()
			print('propagation turn {}, {} changes'.format(turn, changes))
			turn += 1
