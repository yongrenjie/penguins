from pathlib import Path

import penguins as pg
import matplotlib.pyplot as plt


datadir = Path(__file__).parent / "data"

# Test 2D data
if False:
    spec2d = pg.read(datadir / "exam2d_HC", 3, 1)
    spec2d.stage(levels=(5e4, None, None),
                 bounds=((133.8, 50), (7.4, 4)),
                 )
    spec2d.stage(levels=(1e6, None, None),
                 bounds=((50, 6.8), (4, 0.6)),
                 )
    pg.plot()
    pg.show()


# Test offset
if True:
    spec2d = pg.read(datadir / "exam2d_HC", 3, 1)
    spec2d.stage(levels=(1e6, None, None),
                 )
    spec2d.stage(levels=(1e6, None, None),
                 )
    pg.plot(offset=(10, 2))
    # Should see two copies of the same spectrum. Double vision!
    pg.show()

