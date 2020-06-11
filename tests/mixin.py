import penguins as pg

# Test 1D data
spec1d = pg.read("data/1d", 1, 1)
print(spec1d.raw_data())

# Test 1D projection data
spec1dproj = pg.read_abs("data/1dproj/pdata/999")

# Test 2D data
spec2d = pg.read("data/exam2d_HC/", 3, 1)

