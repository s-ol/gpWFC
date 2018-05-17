import pyopencl as cl
import pyopencl.array
import numpy as np

import model
import observe
import propagate
import preview

model = model.Model2d((8, 8,))

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
