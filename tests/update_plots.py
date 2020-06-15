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
    hsqc_dataset.stage(f1_bounds="10..141",
                       f2_bounds="0.5..8.5",
                       colors=("seagreen", "hotpink"),
                       levels=2e4)
    pg.mkplot()
    # pg.show()
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
    hsqc_ds.stage(f1_bounds="11..81",
                  f2_bounds=(1, 4.2),
                  colors=("blue", "red"),
                  )
    pg.mkplot()
    # pg.show()                                   # CONSTRUCT
    pg.savefig("../docs/images/quickstart_plot2d.png", dpi=500)
    print("Done plot b.")


c = all or 0
if c:
    prot = pg.read("data/rot1", 1, 1)
    prot.stage(bounds="..7",
               color="darkviolet",
               label=r"$\mathrm{^{1}H}$")
    pg.mkplot()
    # pg.show()
    pg.savefig("../docs/images/quickstart_plot1d.png", dpi=500)
    print("Done plot c.")


# -- plot1d.rst ------------------------

d = all or 0
if d:
    ds1 = pg.read("data/pt2", 1, 1)
    # This label demonstrates some of the LaTeX capabilities.
    # The colour for this one defaults to the first item in Seaborn/deep.
    ds1.stage(bounds=(7.5, 8.5),
              label=r"$\mathrm{C_{20}H_{28}N_2O_4S}$",
              plot_options={"linestyle": '--'})
    # You can stage the same dataset multiple times with different options.
    ds1.stage(scale=0.2,
              bounds=(8, 8.5),
              label="Yes, that is the actual formula",
              color="hotpink")
    pg.mkplot()
    # pg.show()
    pg.savefig("../docs/images/plot1d_stage.png", dpi=500)
    print("Done plot d.")


e = all or 0
if e:
    ds2 = pg.read("data/pt2", 2, 1)          # 13C spectrum
    ds2.stage(color="black")                 # Full spectrum
    ds2.stage(bounds="100..150")             # Three subspectra
    ds2.stage(bounds="50..100")
    ds2.stage(bounds="0..50")
    pg.mkplot(stacked=True, title="stacked")   # Either this...
    # pg.show()
    pg.savefig("../docs/images/plot1d_stacked.png", dpi=500)

    ds2.stage(color="black")                 # Full spectrum
    ds2.stage(bounds="100..150")             # Three subspectra
    ds2.stage(bounds="50..100")
    ds2.stage(bounds="0..50")
    pg.mkplot(voffset=1.1, title="voffset")    # ...or this
    # pg.show()
    pg.savefig("../docs/images/plot1d_voffset.png", dpi=500)
    print("Done plot e.")



# -- plot2d.rst ------------------------

f = all or 0
if f:
    d = pg.read("data/pt2", 5, 1)   # HMBC
    # Split spectrum into four portions
    bottom_f1, top_f1 = "100..", "..100"
    left_f2, right_f2 = "4.5..", "..4.5"
    # To make this less boring you could use a double listcomp or
    # itertools.product(), but for now we'll do it the repetitive way.
    # Recall levels=1e2 is the same as levels=(1e2, None, None).
    d.stage(f1_bounds=bottom_f1, f2_bounds=left_f2,  levels=1e2)
    d.stage(f1_bounds=top_f1,    f2_bounds=left_f2,  levels=1e3)
    d.stage(f1_bounds=bottom_f1, f2_bounds=right_f2, levels=1e4)
    d.stage(f1_bounds=top_f1,    f2_bounds=right_f2, levels=1e5)
    # Construct and display
    pg.mkplot()
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
                f1_bounds=(11, 80),
                f2_bounds=(0.6, 4.2),
                levels=2.8e5,
                label=f"{temp} K")
    # Separate each plot a little bit
    pg.mkplot(offset=(0.2, 0.02), legend_loc="upper left")
    # pg.show()
    pg.savefig("../docs/images/plot2d_offset.png", dpi=500)
    print("Done plot g.")



# -- cookbook.rst ----------------------

h = all or 0
if h:
    ds = pg.read("data/pt2", 1, 1)
    ds.stage(bounds="7..8.5")
    _, ax = pg.mkplot()
    ax.text(x=7.27, y=0.8, s=r"$\mathrm{CHCl_3}$",
            color="red",
            transform=ax.get_xaxis_transform())
    # pg.show()
    pg.savefig("../docs/images/cookbook_text.png", dpi=500)
    print("Done plot h.")


ia = all or 0
if ia:
    noes = [pg.read("data/rot1", i, 1) for i in range(10, 15)]
    for noe in noes:
        mixing_time = int(noe["d8"] * 1000)
        noe.stage(label=f"{mixing_time} ms",
                  bounds="0..6")
    pg.mkplot(voffset=0.01, hoffset=0.05)
    # pg.show()
    pg.savefig("../docs/images/cookbook_noesy1.png", dpi=500)
    print("Done plot ia.")

ib = all or 0
if ib:
    noes = [pg.read("data/rot1", i, 1) for i in range(10, 15)]
    # Calculate the height of the intense peak
    maxheight = np.amax(noes[0].proc_data())
    # Construct a filtering function
    not_too_tall = lambda i: i < 0.02 * maxheight
    for noe in noes:
        mixing_time = int(noe["d8"] * 1000)
        noe.stage(label=f"{mixing_time} ms",
                  bounds="0..6",
                  dfilter=not_too_tall,
                  scale=-1)
    pg.mkplot(voffset=0.4, hoffset=0.05)
    # pg.show()
    pg.savefig("../docs/images/cookbook_noesy2.png", dpi=500)
    print("Done plot ib.")


ic = all or 0
if ic:
    noes = [pg.read("data/rot1", i, 1) for i in range(10, 15)]
    for noe in noes:
        mixing_time = int(noe["d8"] * 1000)
        noe.stage(label=f"{mixing_time} ms",
                  bounds="0..6",
                  scale=-1)
    _, ax = pg.mkplot(voffset=0.01, hoffset=0.05)
    ax.set_xlim(6.2, -0.3)   # must be (larger, smaller)
    ax.set_ylim(-2.1e4, 1.4e5)
    # pg.show()
    pg.savefig("../docs/images/cookbook_noesy3.png", dpi=500)
    print("Done plot ic.")


id = all or 0
if id:
    noes = [pg.read("data/rot1", i, 1) for i in range(10, 15)]
    for noe in noes:
        noe.stage(bounds="0..6", scale=-1)
    _, ax = pg.mkplot(voffset=0.01, hoffset=0.05)
    ax.set_xlim(6.2, -0.3)   # must be (larger, smaller)
    ax.set_ylim(-2.1e4, 1.4e5)
    # Get the heights of each spectrum, in data coordinates
    heights = pg.pgplot.get_properties().heights
    for height, noe in zip(heights, noes):
        mixing_time_label = f"{int(noe['d8'] * 1000)} ms"
        ax.text(x=0.6, y=height,
                s=mixing_time_label)
    # pg.show()
    pg.savefig("../docs/images/cookbook_noesy4.png", dpi=500)
    print("Done plot id.")


ie = all or 0
if ie:
    noes = [pg.read("data/rot1", i, 1) for i in range(10, 15)]
    for noe in noes:
        noe.stage(bounds="0..6", scale=-1)
    _, ax = pg.mkplot(voffset=0.01, hoffset=0.05)
    ax.set_xlim(6.2, -0.3)   # must be (larger, smaller)
    ax.set_ylim(-2.1e4, 1.4e5)
    # Get the properties of each spectrum
    heights = pg.pgplot.get_properties().heights
    colors = pg.pgplot.get_properties().colors
    for n, (color, height, noe) in enumerate(zip(colors, heights, noes)):
        mixing_time_label = f"{int(noe['d8'] * 1000)} ms"
        ax.text(x=(0.6 - n * 0.05), y=height+2e3,
                s=mixing_time_label,
                color=color)
    # pg.show()
    pg.savefig("../docs/images/cookbook_noesy5.png", dpi=500)
    print("Done plot ie.")


# -- make the docs ---------------------
make_docs = all or 0
if make_docs:
    import os
    os.chdir(os.path.abspath("../docs"))
    os.system("make clean && make html")
