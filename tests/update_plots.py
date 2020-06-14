# This script generates all the images for the documentation.

# In vim, use C-X to disable, C-A to enable

import penguins as pg
import numpy as np

all = 0
if all:
    print("'all' turned on.")

# -- index.rst -------------------------

a = all or 0
if a:
    hsqc_dataset = pg.read("data/pt2", 4, 1)
    hsqc_dataset.stage(f1_bounds=(141, 10),
                       f2_bounds=(8.5, 0.5),
                       colors=("seagreen", "hotpink"),
                       levels=(2e4, None, None)
                       )
    pg.plot()
    pg.savefig("../docs/images/splash_hsqc.png", dpi=500)
    print("Done plot a.")


# -- quickstart.rst --------------------

b = all or 0
if b:
    hsqc_ds = pg.read("data/rot1", 3, 1)
    assert hsqc_ds["ns"] == 16
    assert np.all(hsqc_ds["td"] == np.array([128, 2048]))
    assert np.all(hsqc_ds["si"] == np.array([1024, 2048]))
    assert hsqc_ds["nuc1"] == ("13C", "1H")
    hsqc_ds.stage(f1_bounds=(81, 11),
                  f2_bounds=(4.2, 1),  # STAGING
                  colors=("blue", "red"),
                  levels=(5e4, None, None)
                  )
    pg.plot()                                   # CONSTRUCT
    # pg.show()
    pg.savefig("../docs/images/quickstart_plot2d.png", dpi=500)
    print("Done plot b.")


c = all or 0
if c:
    prot = pg.read("data/rot1", 1, 1)
    prot.stage(bounds=(7, None),         # no right bound
               color="darkviolet",
               label=r"$\mathrm{^{1}H}$ spectrum")
    pg.plot()
    # pg.show()
    pg.savefig("../docs/images/quickstart_plot1d.png", dpi=500)
    print("Done plot c.")


# -- plot1d.rst ------------------------

d = all or 0
if d:
    ds1 = pg.read("data/pt2", 1, 1)
    # This label demonstrates some of the LaTeX capabilities.
    # The colour for this one defaults to the first item in Seaborn/deep.
    ds1.stage(bounds=(8.5, 7.5),
              label=r"$\mathrm{C_{20}H_{28}N_2O_4S}$",
              plot_options={"linestyle": '--'})

    # You can stage the same dataset multiple times with different options.
    ds1.stage(scale=0.2,
              bounds=(8.5, 8),
              label="Yes, that is the actual formula",
              color="hotpink")

    pg.plot()
    pg.savefig("../docs/images/plot1d_stage.png", dpi=500)
    print("Done plot d.")


e = all or 0
if e:
    ds2 = pg.read("data/pt2", 2, 1)          # 13C spectrum
    ds2.stage(bounds=None, color="black")    # Full spectrum
    ds2.stage(bounds=(150, 100))             # Three subspectra
    ds2.stage(bounds=(100, 50))
    ds2.stage(bounds=(50, 0))
    pg.plot(stacked=True, title="stacked")   # Either this...
    pg.savefig("../docs/images/plot1d_stacked.png", dpi=500)

    ds2.stage(bounds=None, color="black")    # Full spectrum
    ds2.stage(bounds=(150, 100))             # Three subspectra
    ds2.stage(bounds=(100, 50))
    ds2.stage(bounds=(50, 0))
    pg.plot(voffset=1.1, title="voffset")    # ...or this
    pg.savefig("../docs/images/plot1d_voffset.png", dpi=500)
    print("Done plot e.")



# -- plot2d.rst ------------------------

f = all or 1

if f:
    d = pg.read("data/pt2", 5, 1)   # HMBC
    # Split spectrum into four portions
    upper_f1, lower_f1 = (None, 100), (100, None)
    upper_f2, lower_f2 = (None, 4.5), (4.5, None)
    # To make this less boring you could use a double listcomp or
    # itertools.product(), but for now we'll do it the repetitive way.
    # Recall levels=1e2 is the same as levels=(1e2, None, None).
    d.stage(f1_bounds=upper_f1, f2_bounds=upper_f2, levels=1e2)
    d.stage(f1_bounds=lower_f1, f2_bounds=upper_f2, levels=1e3)
    d.stage(f1_bounds=upper_f1, f2_bounds=lower_f2, levels=1e4)
    d.stage(f1_bounds=lower_f1, f2_bounds=lower_f2, levels=1e5)
    # Construct and display
    pg.plot()
    # pg.show()
    pg.savefig("../docs/images/plot2d_baselev.png", dpi=500)
    print("Done plot f.")



g = all or 0

if g:
    d = pg.read("data/rot1", 3, 1)   # HSQC
    # Make some colours 
    temps = [240, 250, 260, 270]  # in K
    blues = [f"#00{cc}ff" for cc in ["00", "55", "a6", "ea"]]
    reds = [f"#ff{cc}00" for cc in ["00", "55", "a6", "ea"]]
    # Stage each of them with different colours
    for temp, blue, red in zip(temps, blues, reds):
        d.stage(colors=(blue, red),
                f1_bounds=(80, 11),
                f2_bounds=(4.2, 0.6),
                levels=2.8e5,
                label=f"{temp} K")
    # Separate each plot a little bit
    pg.plot(offset=(0.2, 0.02), legend_loc="upper left")
    # pg.show()
    pg.savefig("../docs/images/plot2d_offset.png", dpi=500)
    print("Done plot g.")




# -- make the docs ---------------------
make_docs = all or 1
if make_docs:
    import os
    os.chdir(os.path.abspath("../docs"))
    os.system("make clean && make html")
