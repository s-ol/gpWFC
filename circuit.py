from models import Model2d, SpriteTile
from observers import CLObserver
from propagators import CL1Propagator
from previews import SpritePreviewWindow
from runners import BacktrackingRunner
from pyglet import app, image, clock
import sys

model = Model2d((16, 16))
# 0: empty pcb
# 1: masked track
# 2: bridge
# 3/4: component_edge
# 7: component_center

# component tiles
model.add(SpriteTile('tiles/component.png', (7, 7, 7, 7), weight=20))
# model.add_rotations(SpriteTile('tiles/corner.png', (3, 0, 0, 3), weight=10), [0, 1, 2, 3])
# model.add_rotations(SpriteTile('tiles/connection.png', (3, 1, 3, 4), weight=10), [0, 1, 2, 3])
model.add(SpriteTile('tiles/corner.png', (3, 0, 0, 4), weight=10, rotation=0))
model.add(SpriteTile('tiles/corner.png', (3, 4, 0, 0), weight=10, rotation=1))
model.add(SpriteTile('tiles/corner.png', (0, 4, 3, 0), weight=10, rotation=2))
model.add(SpriteTile('tiles/corner.png', (0, 0, 3, 4), weight=10, rotation=3))
model.add(SpriteTile('tiles/connection.png', (3, 1, 3, 7), weight=10, rotation=0))
model.add(SpriteTile('tiles/connection.png', (7, 4, 1, 4), weight=10, rotation=1))
model.add(SpriteTile('tiles/connection.png', (3, 7, 3, 1), weight=10, rotation=2))
model.add(SpriteTile('tiles/connection.png', (1, 4, 7, 4), weight=10, rotation=3))

# bridge tiles
# model.add_rotations(SpriteTile('tiles/bridge.png', (2, 1, 2, 1), weight=1), [0, 1])
# model.add_rotations(SpriteTile('tiles/wire.png', (2, 0, 2, 0), weight=0.5), [0, 1])
# model.add_rotations(SpriteTile('tiles/transition.png', (0, 2, 0, 1), weight=0.4), [0, 1, 2, 3])

# track tiles
# model.add_rotations(SpriteTile('tiles/t.png', (1, 0, 1, 1), weight=1.3), [0, 1, 2, 3])
# model.add_rotations(SpriteTile('tiles/viad.png', (1, 0, 1, 0), weight=0.1), [0, 1])
model.add_rotations(SpriteTile('tiles/track.png', (0, 1, 0, 1), weight=10.0), [0, 1])
# model.add_rotations(SpriteTile('tiles/turn.png', (0, 1, 1, 0), weight=1), [0, 1, 2, 3])
model.add_rotations(SpriteTile('tiles/skew.png', (0, 1, 1, 0), weight=2), [0, 1, 2, 3])
model.add_rotations(SpriteTile('tiles/dskew.png', (1, 1, 1, 1), weight=2), [0, 1])

# model.add_rotations(SpriteTile('tiles/vias.png', (0, 1, 0, 0), weight=0.3), [0, 1, 2, 3])

model.add(SpriteTile('tiles/substrate.png', (0, 0, 0, 0), weight=2))

runner = BacktrackingRunner(model, Propagator=CL1Propagator, Observer=CLObserver)
preview = SpritePreviewWindow(runner, 14)

if 'render' in sys.argv[1:]:
	preview.render()
else:
	preview.launch()
