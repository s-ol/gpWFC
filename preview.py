import pyglet
import pyglet.text
import numpy as np
from pyglet.window import key

class PreviewWindow(pyglet.window.Window):
	def __init__(self, model, queue, observer, propagator):
		super().__init__(width=512, height=512)
		self.model = model
		self.observer = observer
		self.propagator = propagator
		self.grid = self.model.build_grid(queue)
		self.grid_array = self.grid.get()
		self.done = False

		tile = pyglet.resource.image('tile.png')
		tile.anchor_x = 32
		tile.anchor_y = 32
		self.sprite = pyglet.sprite.Sprite(img=tile, x=0, y=0)

	def draw_tiles(self, pos, tiles):
		if len(tiles) == 0:
			return

		x, y  = pos
		self.sprite.x = x * 64 + 32
		self.sprite.y = 512 - y * 64 - 32

		self.sprite.opacity = 255 / len(tiles)

		for tile in tiles:
			for direction, adj in enumerate(tile.adj):
				if adj:
					self.sprite.rotation = direction * 90
					self.sprite.draw()

	def on_draw(self):
		self.clear()
		for pos, bits in np.ndenumerate(self.grid_array):
			tiles = self.model.get_tiles(bits)
			self.draw_tiles(pos, tiles)

	def step(self):
		status = self.observer.observe(self.grid)
		if status[0] == 'continue':
			index, collapsed = status[1:]
			self.propagator.propagate(self.grid, index, collapsed)
		else:
			self.done = True
		self.grid.get(ary=self.grid_array)

	def on_key_press(self, symbol, modifiers):
		if symbol == key.SPACE:
			self.step()
