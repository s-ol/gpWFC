import pyopencl as cl
import pyopencl.array
import numpy as np

import observe
import preview
import propagate

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
		self.weights = None

	def add(self, adj, weight=1):
		self.tiles.append(Tile(adj, weight, len(self.tiles)))

	def build_grid(self, queue):
		all_tiles = sum(tile.flag for tile in self.tiles)
		print('filling grid with {}'.format(all_tiles))
		return cl.array.to_device(queue, np.full(self.world_shape, all_tiles, dtype=cl.cltypes.ulong))

	def get_tiles(self, bits):
		return [tile for tile in self.tiles if tile.flag & bits]

model = Model((8, 8,))

adjs = [0, 1, 2]
for adj in np.stack(np.meshgrid(adjs, adjs, adjs, adjs), -1).reshape(-1, 4):
	bins = np.bincount(adj, minlength=3)
	if bins[0] % 2 == 1:
		continue
	if bins[1] % 2 == 1:
		continue
	# if bins[2] % 2 == 1:
	# 	continue

	model.add(adj)

print('{} tiles:'.format(len(model.tiles)))

if __name__ == '__main__':
	import sys
	from pyglet import app, image, clock

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

	print('using {} on {}'.format(platform.name, device.name))

	observer = observe.Observer(ctx, queue, model)
	propagator = propagate.CL1Propagator(ctx, queue, model)
	preview = preview.PreviewWindow(model, queue, observer, propagator)

	iteration = 0
	if 'render' in sys.argv[1:]:
		def screenshot():
			global iteration
			image.get_buffer_manager().get_color_buffer().save('shots/{:04}.png'.format(iteration))
			iteration += 1

		while not preview.done:
			clock.tick()

			preview.switch_to()
			preview.dispatch_events()
			preview.dispatch_event('on_draw')
			preview.flip()
			screenshot()
			preview.step()
		screenshot()
	else:
		app.run()
