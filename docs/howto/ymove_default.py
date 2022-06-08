import numpy as np
import matplotlib.pyplot as plt
import penguins as pg

ds = pg.read(".", 2)
ds.stage(levels=5e5, f1_bounds="0.3..7", f2_bounds="0.3..7")
pg.mkplot()

pg.ymove()
pg.show()
