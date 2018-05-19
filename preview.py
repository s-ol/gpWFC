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
		self.colors = ( (0, 0, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255) )
		self.rotations = [0, 90, 180, 270]

	def draw_tiles(self, pos, bits):
		if bits == 0:
			return

		x, y = pos[-2:]
		self.sprite.x = x * 64 + 32
		self.sprite.y = 512 - y * 64 - 32

		tiles = self.model.get_allowed_tiles(bits)
		self.sprite.opacity = 255 / len(tiles)

		for tile in tiles:
			for direction, adj in enumerate(tile.adj):
				if adj < 1 or self.rotations[direction] == None:
					continue
				self.sprite.color = self.colors[adj]
				self.sprite.rotation = self.rotations[direction] 
				self.sprite.draw()

		if self.debug:
			pyglet.text.Label(str(bits), x=self.sprite.x, y=self.sprite.y).draw()

	def on_draw(self):
		self.clear()

		for pos, bits in np.ndenumerate(self.grid_array):
			self.draw_tiles(pos, bits)

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

class SpritePreviewWindow(PreviewWindow):
	def __init__(self, model, queue, observer, propagator, tile_size):
		super().__init__(model, queue, observer, propagator)
		self.tile_size = tile_size

	def draw_tiles(self, pos, bits):
		if bits == 0:
			return

		x, y = pos[-2:]
		self.sprite.x = x * self.tile_size + self.tile_size/2
		self.sprite.y = 512 - y * self.tile_size - self.tile_size/2

		tiles = self.model.get_allowed_tiles(bits)
		self.sprite.opacity = 255 / len(tiles)

		for tile in tiles:
			tile.image.anchor_x = self.tile_size/2
			tile.image.anchor_y = self.tile_size/2
			self.sprite.image = tile.image
			self.sprite.rotation = tile.rotation * 90
			self.sprite.draw()

		if self.debug:
			pyglet.text.Label(str(bits), x=self.sprite.x, y=self.sprite.y).draw()

class PreviewWindow3d(PreviewWindow):
	def __init__(self, *args):
		super().__init__(*args)
		self.rotations = [0, 90, None, 180, 270, None]
		self.slice = 0

	def on_draw(self):
		self.clear()

		for pos, bits in np.ndenumerate(self.grid_array[self.slice]):
			self.draw_tiles(pos, bits)

	def on_key_press(self, symbol, modifiers):
		if symbol == key.UP:
			self.slice += 1
		elif symbol == key.DOWN:
			self.slice += 1
		else:
			super().on_key_press(symbol, modifiers)
			return
		self.slice = self.slice % len(self.grid_array)
		print(self.slice)
