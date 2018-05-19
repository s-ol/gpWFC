import pyopencl as cl
import pyopencl.cltypes
import numpy as np

class Tile(object):
	nextIndex = 0
	def __init__(self, adj, weight, index):
		self.adj = adj
		self.weight = weight
		self.index = index
		self.flag = np.uint64(1 << self.index)

	def compatible(self, other, direction):
		l = len(self.adj)
		return self.adj[direction] == other.adj[(direction+l//2) % l]

class Model(object):
	def __init__(self, world_shape):
		self.tiles = []
		self.world_shape = world_shape

	def add(self, adj, weight=1):
		self.tiles.append(Tile(adj, weight, len(self.tiles)))

	def build_grid(self, queue):
		all_tiles = sum(tile.flag for tile in self.tiles)
		print('filling grid with {}'.format(all_tiles))
		return cl.array.to_device(queue, np.full(self.world_shape, all_tiles, dtype=cl.cltypes.ulong))

	def get_allowed_tiles(self, bits):
		return [tile for tile in self.tiles if tile.flag & bits]

class Model2d(Model):
	adjacent = 4

	def __init__(self, world_shape):
		assert len(world_shape) == 2
		super().__init__(world_shape)

	def get_neighbours(self, pos):
		w, h = self.world_shape
		x, y = pos
		yield (x-1)%w, y
		yield x, (y-1)%h
		yield (x+1)%w, y
		yield x, (y+1)%h

class Model3d(Model):
	adjacent = 6

	def __init__(self, world_shape):
		assert len(world_shape) == 3
		super().__init__(world_shape)

	def get_neighbours(self, pos):
		w, h, d = self.world_shape
		x, y, z = pos
		yield (x-1)%w, y, z
		yield x, (y-1)%h, z
		yield x, y, (z-1)%d
		yield (x+1)%w, y, z
		yield x, (y+1)%h, z
		yield x, y, (z+1)%d
