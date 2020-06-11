from pathlib import Path

import penguins as pg
import matplotlib.pyplot as plt


datadir = Path(__file__).parent / "data"

# Test 2D data
spec2d = pg.read(datadir / "exam2d_HC", 3, 1)
# plt.figure(figsize=(7,5))
pg.plot2d(spec2d,
          contour_levels=(1e6, None, None),
          contour_colors=("green", "purple"),
          bounds=((133.8, 50), (7.4, 4)),
          )
pg.plot2d(spec2d,
          contour_levels=(1e6, None, None),
          contour_colors=("blue", "red"),
          bounds=((50, 6.8), (4, 0.6)),
          )
plt.show()


# Test 1D data
spec1d = pg.read(datadir / "1d", 1, 1)
spec1d.plot(label="Quack?",
            plot_options={"color": "darkviolet"})
plt.show()


# Test 1D projection data
spec1dproj = pg.read_abs(datadir / "1dproj" / "pdata" / "999")
pg.plot1d(spec1dproj,
          label="Hello there!",
          bounds=(None, 4))    # from left edge to 4 ppm
pg.plot1d(spec1dproj,
          label="Hello THERE!",
          bounds=(4, None))    # from 4 ppm to right edge
plt.show()
