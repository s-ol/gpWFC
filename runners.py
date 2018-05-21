from pyopencl import create_some_context, CommandQueue
from pyopencl.array import to_device
from observers import CLObserver
from propagators import CL1Propagator

class Runner(object):
	def __init__(self, model, Observer=CLObserver, Propagator=CL1Propagator, ctx=None):
		if not ctx:
			ctx = create_some_context()
		self.model = model

		self.grid_array = self.model.build_grid()
		with CommandQueue(ctx) as queue:
			self.grid = to_device(queue, self.grid_array)
		self.observer = Observer(model, ctx=ctx)
		self.propagator = Propagator(model, ctx=ctx)

		self.candidate = self.observer.observe(self.grid)[1:]
		self.done = False

	def step(self):
		index, collapsed = self.candidate
		self.propagator.propagate(self.grid, index, collapsed)
		status = self.observer.observe(self.grid)

		if status[0] == 'continue':
			self.candidate = status[1:]
		elif status[0] == 'error':
			self.done = True
		else:
			self.grid.get(ary=self.grid_array)

		return status[0]

	def run(self):
		while not self.done:
			yield self.step()

	def finish(self):
		status = None
		for status in self.run():
			pass
		return status

class BacktrackingRunner(Runner):
	def __init__(self, *args, snapshot_every=4, **kwargs):
		super().__init__(*args, **kwargs)
		self.snapshot = self.grid_array.copy()
		self.snapshot_every = snapshot_every
		self.snapshot_age = 0

	def step(self):
		index, collapsed = self.candidate
		self.propagator.propagate(self.grid, index, collapsed)
		status = self.observer.observe(self.grid)
		self.grid.get(ary=self.grid_array)

		if status[0] == 'continue':
			self.candidate = status[1:]
			self.snapshot_age += 1
			if self.snapshot_age >= self.snapshot_every:
				self.snapshot = self.grid_array.copy()
				self.snapshot_age = 0
		elif status[0] == 'error':
			if not self.snapshot is None:
				print('backtracking {} rounds'.format(self.snapshot_age))
				self.grid.set(self.snapshot)
				self.candidate = self.observer.observe(self.grid)[1:]
				self.snapshot = None
				self.snapshot_age = self.snapshot_every
				return self.step()
			else:
				print('cannot backtrack anymore')
				self.done = True
		else:
			self.done = True

		return status[0]
