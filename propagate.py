import pyopencl as cl
import pyopencl.array
import pyopencl.cltypes
import numpy as np

def xy2i(x, y):
	x = x % 8
	y = y % 8
#	return y * 8 + x
	return x * 8 + y

def xyt2i(p):
	return xy2i(p[0], p[1])

def i2xy(i):
	x = i // 8
#	return (i % 8), i // 8
	return i //8, (i % 8)

class BasePropagator(object):
	def __init__(self, model):
		self.model = model

		self.neighbours = np.fromfunction(
			np.vectorize(self.get_neighbours),
			self.model.world_shape + (self.model.adjacent,),
			dtype=int
		).astype(cl.cltypes.uint)

		self.allows = np.fromfunction(
			np.vectorize(self.get_allows),
			(len(self.model.tiles), self.model.adjacent),
			dtype=int
		).astype(cl.cltypes.ulong)

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

	def get_config(self):
		adjacent_bits = (self.model.adjacent - 1).bit_length()
		adjacent_pow = 1 << adjacent_bits

		config = {
			'adj_uint': 'uint' + str(adjacent_pow),
			'adj_ulong': 'ulong' + str(adjacent_pow),
			'states': len(self.model.tiles),
			'forNeighbour': lambda tpl, join='\n': join.join([tpl.format(i=i) for i in range(self.model.adjacent)]),
		}

		config['preamble'] = '''
			#define ADJ {adj}
			#define ADJUINT {adj_uint}
			#define ADJULONG {adj_ulong}
			#define STATES {states}
		'''.format(adj=self.model.adjacent, **config)

		return config

class CPUPropagator(BasePropagator):
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
				i2xy(self.neighbours[i + (neighbour,)]), allowmaps[neighbour],
				grid
			)

	def propagate(self, grid, index, collapsed):
		self.reduce_to_allowed(index, np.uint64(collapsed), grid)

class CL2Propagator(BasePropagator):
	def __init__(self, model, ctx):
		super().__init__(model)

		self.ctx = ctx

		alloc = cl.tools.ImmediateAllocator(queue, mem_flags=cl.mem_flags.READ_ONLY)
		self.allows_buf = cl.array.to_device(queue, self.allows, alloc)
		self.neighbours_buf = cl.array.to_device(queue, self.neighbours, alloc)

		config = self.get_config()
		self.program = cl.Program(ctx, config['preamble'] + '''
		__kernel void reduce_to_allowed(
			const uint i, const ulong allowmap,
			__global ulong* grid, __global ADJULONG* allows, __global ADJUINT* neighbours
		) {
			ulong old_bits = grid[i];
			ulong new_bits = old_bits & allowmap;
			grid[i] = new_bits;
			ulong diff = old_bits ^ new_bits;
			if (!diff) return;

			ADJULONG allowmaps;
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

	def propagate(self, grid, index, collapsed):
		with cl.CommandQueue(self.ctx) as queue:
			self.program.reduce_to_allowed(
				queue, (1,), None,
				xyt2i(index), collapsed,
				grid, self.allows_buf, self.neighbours_buf
			)

class CL1Propagator(BasePropagator):
	def __init__(self, model, ctx):
		super().__init__(model)

		queue = cl.CommandQueue(ctx)
		alloc = cl.tools.ImmediateAllocator(queue, mem_flags=cl.mem_flags.READ_ONLY)
		self.allows_buf = cl.array.to_device(queue, self.allows, alloc)
		self.neighbours_buf = cl.array.to_device(queue, self.neighbours, alloc)

		config = self.get_config()
		fN = config['forNeighbour']

		self.update_grid = cl.reduction.ReductionKernel(ctx,
			arguments='__global ulong* grid, __global {adj_ulong}* allows, __global {adj_uint}* neighbours'.format(**config),
			neutral='0',
			dtype_out=cl.cltypes.uint,
			map_expr='update_tile(i, grid, allows, neighbours)',
			reduce_expr='a + b',
			preamble=config['preamble'] + '''
			uint update_tile(uint i, __global ulong* grid, __global ADJULONG* allows, __global ADJUINT* neighbours) {
				ulong old_bits = grid[i];

				ADJUINT next = neighbours[i];
				''' +
				fN('''
					ulong grid_{i} = grid[next.s{i}];
					ulong mask_{i} = 0;
				''') + '''

				for (uint tile = 0; tile < STATES; tile++) {
					ulong flag = 1 << tile;
					ADJULONG tile_allows = allows[tile];
					''' + fN('''
					if (flag & grid_{i}) mask_{i} |= tile_allows.s{i};
					''') + '''
				}

				ulong new_bits = old_bits ''' + fN('& mask_{i}', '') + ''';
				grid[i] = new_bits;
				return old_bits != new_bits;
			}
                        '''
		)

	def propagate(self, grid, index, collapsed):
		grid[index] = collapsed
		turn, changes = 0, 1
		while changes > 0:
			changes = self.update_grid(grid, self.allows_buf, self.neighbours_buf).get()
			print('propagation turn {}, {} changes'.format(turn, changes))
			turn += 1
