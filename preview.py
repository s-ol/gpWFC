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
		self.debug = False

		tile = pyglet.resource.image('tile.png')
		tile.anchor_x = 32
		tile.anchor_y = 32
		self.sprite = pyglet.sprite.Sprite(img=tile, x=0, y=0)
		self.colors = ( (255, 0, 0), (0, 255, 0), (0, 0, 255) )

	def draw_tiles(self, pos, bits):
		if bits == 0:
			return

		x, y = pos
		self.sprite.x = x * 64 + 32
		self.sprite.y = 512 - y * 64 - 32

		tiles = self.model.get_tiles(bits)
		self.sprite.opacity = 255 / len(tiles)

		for tile in tiles:
			for direction, adj in enumerate(tile.adj):
				if adj < 1:
					continue
				self.sprite.color = self.colors[adj]
				self.sprite.rotation = direction * 90
				self.sprite.draw()

		if self.debug:
			pyglet.text.Label(str(bits), x=self.sprite.x, y=self.sprite.y).draw()

	def on_draw(self):
		self.clear()

		batch = pyglet.graphics.Batch()
		for pos, bits in np.ndenumerate(self.grid_array):
			self.draw_tiles(pos, bits)
		batch.draw()

	def step(self):
		status = self.observer.observe(self.grid)
		if status[0] == 'continue':
			index, collapsed = status[1:]
			self.propagator.propagate(self.grid, index, collapsed)
		else:
			self.done = True
		self.grid.get(ary=self.grid_array)
	
	def run(self):
		while not self.done:
			self.step()

	def on_key_press(self, symbol, modifiers):
		if symbol == key.ESCAPE:
			self.close()
		elif symbol == key.SPACE:
			self.step()
		elif symbol == key.R:
			self.run()
		elif symbol == key.D:
			self.debug = not self.debug
