from pathlib import Path
import penguins as pg

datadir = Path(__file__).parent / "data"

# Test 1D data
spec1d = pg.read(datadir / "1d", 1, 1)
spec1d.stage(bounds=(None, 3), label="Quack?", color="brown")
spec1d.stage(bounds=(3, None), label="Quack!", color="green")
pg.plot()   # returns fig, ax
pg.show()

# Test 2D data
spec2d = pg.read(datadir / "exam2d_HC", 3, 1)
# plt.figure(figsize=(7,5))
spec2d.stage(levels=(5e4, None, None),
             colors=("green", "purple"),
             bounds=((133.8, 50), (7.4, 4)),
             )
spec2d.stage(levels=(1e6, None, None),
             colors=("blue", "red"),
             bounds=((50, 6.8), (4, 0.6)),
             )
pg.plot()
pg.show()
