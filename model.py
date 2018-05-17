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
		# @TODO 3d
		return self.adj[direction] == other.adj[(direction+2) % 4]

class Model(object):
	def __init__(self, world_shape):
		self.tiles = []
		self.world_shape = world_shape
		self.adjacent = 4

	def add(self, adj, weight=1):
		self.tiles.append(Tile(adj, weight, len(self.tiles)))

	def build_grid(self, queue):
		all_tiles = sum(tile.flag for tile in self.tiles)
		print('filling grid with {}'.format(all_tiles))
		return cl.array.to_device(queue, np.full(self.world_shape, all_tiles, dtype=cl.cltypes.ulong))

	def get_allowed_tiles(self, bits):
		return [tile for tile in self.tiles if tile.flag & bits]
