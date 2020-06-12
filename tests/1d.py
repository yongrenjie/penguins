from pathlib import Path

import penguins as pg

# Staging_1D
if False:
    ds1 = pg.read("/opt/topspin4.0.8/examdata/exam1d_1H", 1, 1)
    # This label demonstrates some of the LaTeX capabilities.
    ds1.stage(bounds=(8.5, 7.5),     # NH protons only
             label=r"Cyclosporin (C$_{62}$H$_{111}$N$_{11}$O$_{12}$)",
             plot_options={"linestyle": '--'})
    # You can stage the same dataset multiple times with different options.
    ds1.stage(scale=0.2,
             bounds=(8, 7),
             label="Smaller cyclosporin",
             color="hotpink")
    pg.plot(); pg.show()


# Constructing_1D
if True:
    ds1 = pg.read("/opt/topspin4.0.8/examdata/exam1d_1H", 1, 1)
    ds1.stage(bounds=None, color="black")   # Full spectrum
    ds1.stage(bounds=(2, 0))                # Four subspectra
    ds1.stage(bounds=(4, 2))
    ds1.stage(bounds=(6, 4))
    ds1.stage(bounds=(8, 6))
    pg.plot(stacked=True, title="stacked")  # Either this...
    # pg.plot(voffset=1.1, title="voffset")   # ...or this
    pg.show()


# Colour test
if False:
    datadir = Path(__file__).parent / "data"
    spec1d = pg.read(datadir / "1d", 1, 1)
    spec1d.stage(bounds=(None, 9), label="Quack")
    spec1d.stage(bounds=(9, 8), label="Quack")
    spec1d.stage(bounds=(8, 7), label="QUack")
    spec1d.stage(bounds=(7, 6), label="QUack")
    spec1d.stage(bounds=(6, 5), label="QUAck")
    spec1d.stage(bounds=(5, 4), label="QUAck")
    spec1d.stage(bounds=(4, 3), label="QUACk")
    spec1d.stage(bounds=(3, 2), label="QUACk")
    spec1d.stage(bounds=(2, 1), label="QUACK")
    spec1d.stage(bounds=(1, None), label="QUACK")
    pg.plot()   # returns fig, ax
    pg.show()


# Test 1D projection data
# spec1dproj = pg.read_abs(datadir / "1dproj" / "pdata" / "999")
# pg.plot1d(spec1dproj,
#           label="Hello there!",
#           bounds=(None, 4))    # from left edge to 4 ppm
# pg.plot1d(spec1dproj,
#           label="Hello THERE!",
#           bounds=(4, None))    # from 4 ppm to right edge
# plt.show()
