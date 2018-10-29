import pyopencl as cl
import pyopencl.cltypes
import numpy as np
import pyglet

class Tile(object):
	def __init__(self, adj, weight=1):
		self.adj = adj
		self.weight = weight

	def rotated(self, rot):
		adj = self.adj[-rot:] + self.adj[:-rot]
		return Tile(adj, self.weight)

	def register(self, index):
		self.index = index
		self.flag = np.uint64(1 << self.index)

	def compatible(self, other, direction):
		l = len(self.adj)
		return self.adj[direction] == other.adj[(direction+l//2) % l]

class SpriteTile(Tile):
	def __init__(self, image, adj, weight=1, rotation=0):
		super().__init__(adj, weight)
		if not isinstance(image, pyglet.image.AbstractImage):
			image = pyglet.resource.image(image)
		self.image = image
		self.rotation = rotation

	def rotated(self, rotation):
		adj = self.adj[-rotation:] + self.adj[:-rotation]
		return SpriteTile(self.image, adj, weight=self.weight, rotation=rotation)

class Model(object):
	def __init__(self, world_shape):
		self.tiles = []
		self.world_shape = world_shape

	def add(self, tile):
		tile.register(len(self.tiles))
		self.tiles.append(tile)

	def add_rotations(self, orig, rotations):
		for rot in rotations:
			tile = orig.rotated(rot)
			tile.register(len(self.tiles))
			self.tiles.append(tile)

	def build_grid(self):
		all_tiles = sum(tile.flag for tile in self.tiles)
		print('filling grid with {}'.format(all_tiles))
		return np.full(self.world_shape, all_tiles, dtype=cl.cltypes.ulong)

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
