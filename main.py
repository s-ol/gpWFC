import numpy as np

from gpWFC.models import Model2d, Model3d, Tile
from gpWFC.observers import CLObserver
from gpWFC.propagators import CPUPropagator, CL1Propagator
from gpWFC.previews import PreviewWindow, PreviewWindow3d
from gpWFC.runners import BacktrackingRunner

if __name__ == '__main__':
	import sys

	if '3d' in sys.argv[1:]:
		model = Model3d((4, 4, 2))
		model.add(Tile((0, 1, 1, 0, 1, 0))) # all green
		model.add(Tile((2, 0, 0, 2, 0, 1))) # all green
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

	Propagator = CL1Propagator
	if 'cpu' in sys.argv[1:]:
		Propagator = CPUPropagator

	runner = BacktrackingRunner(model, Observer=CLObserver, Propagator=Propagator)

	if 'silent' in sys.argv[1:]:
		from timeit import default_timer

		start = default_timer()
		status = runner.finish()
		print('{} after {}s'.format(status, default_timer() - start))
	else:
		Preview = PreviewWindow
		if '3d' in sys.argv[1:]:
			Preview = PreviewWindow3d

		preview = Preview(runner)
		if 'render' in sys.argv[1:]:
			preview.render()
		else:
			preview.launch()
