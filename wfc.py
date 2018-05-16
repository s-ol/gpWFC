import pyopencl as cl
import pyopencl.array
import numpy as np

import observe
import preview
import propagate

platform = None
platforms =  cl.get_platforms()
if len(platforms) == 0:
	platform = platforms[0]
for pl in platforms:
	if 'NVIDIA' in pl.name:
		platform = pl

device = platform.get_devices()[0]
print(device)
ctx = cl.Context([ device ])
queue = cl.CommandQueue(ctx)

print('using {} CL platform'.format(platform.name))

class Tile(object):
	nextIndex = 0
	def __init__(self, char, adj, weight, index):
		self.char = char
		self.adj = adj
		self.weight = weight
		self.index = index
		self.flag = 1 << self.index

	def compatible(self, other, direction):
		# @TODO 3d
		return self.adj[direction] == other.adj[(direction+2) % 4]


class Model(object):
	def __init__(self, world_shape):
		self.tiles = []
		self.world_shape = world_shape
		self.weights = None

	def add(self, char, adj, weight=1):
		self.tiles.append(Tile(char, adj, weight, len(self.tiles)))

	def finish(self):
		self.weights = cl.array.to_device(queue, np.array(list(tile.weight for tile in self.tiles), dtype=cl.cltypes.float))

	def build_grid(self):
		all_tiles = sum(tile.flag for tile in self.tiles)
		print('filling grid with {}'.format(all_tiles))
		return cl.array.to_device(queue, np.full(self.world_shape, all_tiles, dtype=cl.cltypes.uint))

	def get_tiles(self, bits):
		return [tile for tile in self.tiles if tile.flag & bits]

model = Model((8, 8,))
# left, up, right, down
model.add('┃', [0, 1, 0, 1])
model.add('━', [1, 0, 1, 0])
model.add('┓', [1, 0, 0, 1])
model.add('┗', [0, 1, 1, 0])
model.add('┳', [1, 0, 1, 1])
model.add('┻', [1, 1, 1, 0])
model.add('╹', [0, 1, 0, 0])
model.add('╻', [0, 0, 0, 1])
model.finish()

observer = observe.WFCObserver(ctx, queue, model)
propagator = propagate.WFCPropagator(ctx, queue, model)
preview = preview.PreviewWindow(model, observer, propagator)

import pyglet
pyglet.app.run()
