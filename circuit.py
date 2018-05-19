from model import Model2d, SpriteTile
from observe import Observer
from propagate import CL1Propagator
from preview import SpritePreviewWindow
from pyglet import app, image, clock
from pyopencl import create_some_context, CommandQueue
import sys

model = Model2d((32, 32))
# 0: empty pcb
# 1: masked track
# 2: bridge
# 3: component_edge
# 4: component_center

model.add(SpriteTile('tiles/component.png', (4, 4, 4, 4), weight=20))
model.add_rotations(SpriteTile('tiles/corner.png', (3, 0, 0, 3), weight=10), [0, 1, 2, 3])
model.add_rotations(SpriteTile('tiles/connection.png', (3, 1, 3, 4), weight=10), [0, 1, 2, 3])

model.add(SpriteTile('tiles/substrate.png', (0, 0, 0, 0), weight=2))
# model.add_rotations(SpriteTile('tiles/bridge.png', (2, 1, 2, 1), weight=1), [0, 1])
# model.add_rotations(SpriteTile('tiles/wire.png', (2, 0, 2, 0), weight=0.5), [0, 1])
# model.add_rotations(SpriteTile('tiles/transition.png', (0, 2, 0, 1), weight=0.4), [0, 1, 2, 3])
model.add_rotations(SpriteTile('tiles/t.png', (1, 0, 1, 1), weight=0.1), [0, 1, 2, 3])
model.add_rotations(SpriteTile('tiles/track.png', (0, 1, 0, 1), weight=2.0), [0, 1])
model.add_rotations(SpriteTile('tiles/turn.png', (0, 1, 1, 0), weight=1), [0, 1, 2, 3])
model.add_rotations(SpriteTile('tiles/viad.png', (1, 0, 1, 0), weight=0.1), [0, 1])
# model.add_rotations(SpriteTile('tiles/vias.png', (0, 1, 0, 0), weight=0.3), [0, 1, 2, 3])
model.add_rotations(SpriteTile('tiles/skew.png', (0, 1, 1, 0), weight=2), [0, 1, 2, 3])
model.add_rotations(SpriteTile('tiles/dskew.png', (1, 1, 1, 1), weight=2), [0, 1])

ctx = create_some_context()
device = ctx.devices[0]
queue = CommandQueue(ctx)
observer = Observer(ctx, queue, model)
propagator = CL1Propagator(model, ctx)
preview = SpritePreviewWindow(model, queue, observer, propagator, 14)

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
	screenshot()
else:
	app.run()
