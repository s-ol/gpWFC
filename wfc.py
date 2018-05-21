import pyopencl as cl
import pyopencl.array
import numpy as np

from model import Model2d, Model3d, Tile
import observe
import propagate
import preview

if __name__ == '__main__':
	import sys

	if '3d' in sys.argv[1:]:
		model = Model3d((4, 4, 4))
		model.add((2, 2, 2, 2, 2, 2)) # all blue
		model.add((1, 1, 2, 1, 1, 2)) # blue only above and below
	else:
		model = Model2d((8, 8))
		adjs = [0, 1, 2]
		for adj in np.stack(np.meshgrid(adjs, adjs, adjs, adjs), -1).reshape(-1, 4):
			bins = np.bincount(adj, minlength=3)
			if bins[0] % 2 == 1:
				continue
			if bins[1] % 2 == 1:
				continue
			# if bins[2] % 2 == 1:
			# 	continue
			model.add(Tile(adj))

	print('{} tiles:'.format(len(model.tiles)))


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

	if 'cpu' in sys.argv[1:]:
		propagator = propagate.CPUPropagator(model)
	else:
		propagator = propagate.CL1Propagator(model, ctx)

	if 'silent' in sys.argv[1:]:
		from timeit import default_timer

		grid = model.build_grid(queue)
		start = default_timer()
		while True:
			status = observer.observe(grid)
			if status[0] == 'continue':
				index, collapsed = status[1:]
				propagator.propagate(grid, index, collapsed)
			else:
				break
		print('{} after {}s'.format(status[0], default_timer() - start))
	else:
		from pyglet import app, image, clock
		if '3d' in sys.argv[1:]:
			preview = preview.PreviewWindow3d(model, queue, observer, propagator)
		else:
			preview = preview.PreviewWindow(model, queue, observer, propagator)
		if 'render' in sys.argv[1:]:
			iteration = 0
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
			preview.dispatch_event('on_draw')
			preview.flip()
			screenshot()
		else:
			app.run()
