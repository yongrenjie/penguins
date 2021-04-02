"""
plotutils.py
------------

High-level plotting utilities that (attempt to be / are) broadly useful to a
wide range of figures.
"""

import warnings
from typing import (Any, Sequence, Tuple)

import numpy as np  # type: ignore
import matplotlib
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.ticker import AutoMinorLocator  # type: ignore

from .exportdeco import export


@export
def style_axes(ax: Any,
               style: str,
               tight_layout: bool = True,
               ) -> None:
    """Styles the given |Axes| instance according to the requested style.

    This is useful for making sure that all subplots in a series have a uniform
    appearance.

    Parameters
    ----------
    ax : Axes
        |Axes| instance to style.
    style : str
        Style to be applied. The available options are ``1d``, ``1d_box``,
        ``2d``, ``plot``, and ``natural``.
    tight_layout : bool, default True
        Whether to call |tight_layout| after completion.

    Returns
    -------
    None
    """
    def disable_y_axis(ax):
        ax.yaxis.set_visible(False)

    def remove_top_left_right_spines(ax):
        for s in ["top", "left", "right"]:
            ax.spines[s].set_visible(False)

    def set_xaxis_ticks(ax):
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which="both", width=1.3)
        ax.tick_params(which="major", length=5)
        ax.tick_params(which="minor", length=3)

    def set_xyaxis_ticks(ax):
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which="both", width=1.3)
        ax.tick_params(which="major", length=5)
        ax.tick_params(which="minor", length=3)

    def thicken_spines(ax):
        for s in ["top", "left", "right", "bottom"]:
            ax.spines[s].set_visible(True)
        for s in ["top", "left", "right", "bottom"]:
            ax.spines[s].set_linewidth(1.3)

    if style == "1d":
        disable_y_axis(ax)
        set_xaxis_ticks(ax)
        thicken_spines(ax)
        remove_top_left_right_spines(ax)
    elif style == "1d_box":
        disable_y_axis(ax)
        set_xaxis_ticks(ax)
        thicken_spines(ax)
    elif style == "2d":
        thicken_spines(ax)
        set_xyaxis_ticks(ax)
    elif style == "plot":
        thicken_spines(ax)
    elif style == "natural":
        pass
    else:
        warnings.warn(f"Invalid style '{style}' requested.")

    if tight_layout:
        plt.tight_layout()


@export
def cleanup_axes() -> None:
    # Need to draw the figure to get the renderer.
    fig = plt.gcf()
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    # Iterate over axes and check which ticks overlap with axes label.
    for ax in fig.axes:
        xlabel_bbox = ax.xaxis.label.get_window_extent(renderer=r)
        ylabel_bbox = ax.yaxis.label.get_window_extent(renderer=r)
        # Just check every bbox. label1 and label2 correspond to the bottom and
        # top positions (for x-axis) and the left and right positions (for
        # y-axis).
        for xtick in ax.xaxis.get_major_ticks():
            xtick_bbox1 = xtick.label1.get_window_extent(renderer=r)
            xtick_bbox2 = xtick.label2.get_window_extent(renderer=r)
            if xtick_bbox1.overlaps(xlabel_bbox):
                xtick.label1.set_visible(False)
            if xtick_bbox2.overlaps(xlabel_bbox):
                xtick.label2.set_visible(False)
        for ytick in ax.yaxis.get_major_ticks():
            ytick_bbox1 = ytick.label1.get_window_extent(renderer=r)
            ytick_bbox2 = ytick.label2.get_window_extent(renderer=r)
            if ytick_bbox1.overlaps(ylabel_bbox):
                ytick.label1.set_visible(False)
            if ytick_bbox2.overlaps(ylabel_bbox):
                ytick.label2.set_visible(False)


@export
def cleanup_figure(padding: float = 0.02
                   ) -> None:
    # Resize subplots so that their titles don't clash with the figure legend,
    # or figure suptitle.
    plt.tight_layout()
    fig = plt.gcf()
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    # Get minimum bbox y-extent of figure legend(s).
    inv = fig.transFigure.inverted()
    legend_bboxs = [inv.transform(legend.get_window_extent(renderer=r))
                    for legend in fig.legends]
    # Try to add figure suptitle.
    if fig._suptitle is not None:
        title_bbox = inv.transform(fig._suptitle.get_window_extent(renderer=r))
        legend_bboxs.append(title_bbox)
    if len(legend_bboxs) == 0:
        raise ValueError("cleanup_figure(): no figure legends or titles"
                         " were found")
    legend_miny = min(bbox[0][1] for bbox in legend_bboxs)

    # Find maximum bbox y-extent of axes titles.
    titles = [ax.title for ax in fig.axes]
    title_bboxs = [inv.transform(title.get_window_extent(renderer=r))
                   for title in titles]
    title_yextents = [title_bbox[1][1] - title_bbox[0][1]
                      for title_bbox in title_bboxs]
    max_title_height = max(title_yextents, default=0)

    axes_maxy = legend_miny - padding - max_title_height
    # Resize
    plt.subplots_adjust(top=axes_maxy)


@export
def xmove(ax: Any,
          pos: str,
          remove_ticks: int = 0,
          tight_layout: bool = True,
          ) -> None:
    if pos == "right":
        # Move the label
        ax.xaxis.label.set_horizontalalignment("center")
        ax.xaxis.label.set_verticalalignment("top")
        ax.xaxis.set_label_coords(1.02, -0.05)
    else:
        raise ValueError(f"Invalid position '{pos}' provided.")

    if tight_layout:
        plt.tight_layout()


@export
def ymove(ax: Any,
          pos: str,
          tight_layout: bool = True,
          dx: float = 0,
          dy: float = 0,
          ) -> None:
    """
    Utility function which moves the Axes y-label and y-axis ticks to one of
    several preset configurations. It is a good idea to call `cleanup_axes()`
    after using this function in order to remove any tick labels that clash
    with the axis label.

    Parameters
    ----------
    ax : |Axes|
        The Axes instance to apply the changes to.
    pos : str from {"topright", "midright", "topspin"}
        The configuration of the y-axis. This is better explained through a
        picture than in words.
    tight_layout : bool, default True
        Whether to call |tight_layout| after completion.
    dx : float, default 0
        Amount to horizontally shift the resulting y-axis label by, expressed
        in terms of Axes coordinates (i.e. 0 is left-most part of Axes and 1 is
        right-most). Positive numbers shift the label to the right and vice
        versa. This should generally not be needed but can be useful for
        tweaking the layout if the result is not satisfactory.
    dy : float, default 0
        Amount to vertically shift the resulting y-axis label by, expressed in
        terms of Axes coordinates. Positive numbers shift the label up.
    """
    def move_yticks_to_right():
        ax.yaxis.tick_right()

    def draw_fig():
        fig = plt.gcf()
        fig.canvas.draw()
        return fig.canvas.get_renderer()

    if pos == "topright":
        move_yticks_to_right()
        ax.yaxis.label.set_rotation(0)  # right way up
        ax.yaxis.label.set_horizontalalignment("left")
        ax.yaxis.label.set_verticalalignment("top")
        ax.yaxis.set_label_coords(1.03 + dx, 1 + dy)

    elif pos == "midright":
        move_yticks_to_right()

        # First, figure out the maximum x-extent of ytick labels. We need only
        # consider those within the y-axis limits.
        x_extents = []
        r = draw_fig()
        inv = ax.transAxes.inverted()
        y1, y2 = ax.get_ylim()
        ymin, ymax = min(y1, y2), max(y1, y2)  # account for inverted axis
        for ytick in ax.yaxis.get_major_ticks():
            loc = ytick.get_loc()
            if loc >= ymin and loc <= ymax:
                bbox = ytick.label2.get_window_extent(renderer=r)
                x_extents.append(inv.transform(bbox)[1, 0])
        ax.yaxis.set_label_coords(max(x_extents, default=1) + 0.05 + dx,
                                  0.5 + dy)
        ax.yaxis.label.set_rotation(90)
        ax.yaxis.label.set_horizontalalignment("center")
        ax.yaxis.label.set_verticalalignment("center")

    elif pos == "topspin":
        move_yticks_to_right()
        for ytick in ax.yaxis.get_major_ticks():
            ytick.label2.set_rotation(90)
        # The y-axis label should be bottom-aligned together with these.
        r = draw_fig()
        inv = ax.transAxes.inverted()
        bbox = ytick.label2.get_window_extent(renderer=r)
        x_base = inv.transform(bbox)[1, 0]
        ax.yaxis.label.set_horizontalalignment("right")
        ax.yaxis.label.set_verticalalignment("bottom")
        ax.yaxis.set_label_coords(x_base + dx, 1 + dy)

    else:
        raise ValueError(f"Invalid position '{pos}' provided.")

    if tight_layout:
        plt.tight_layout()


def label_generator(form):
    """
    Internal generator which spits out abcdefghijkl... or 1234567.... or worse,
    Roman numerals.
    """
    chars = "abcdefghijklmnopqrstuvwxyz"
    roman = ("i ii iii iv v vi vii viii ix x xi xii xiii xiv xv xvi"
             " xvii xviii xix xx xxi xxii xxiii xxiv xxv xxvi")
    if form == "a":
        while True:
            yield from chars
    elif form == "A":
        while True:
            yield from chars.upper()
    elif form == "i":
        while True:
            yield from roman.split()
    elif form == "I":
        while True:
            yield from roman.upper().split()
    elif form == "1":
        count = 1
        while True:
            yield count
            count = count + 1


@export
def label_axes(axs: Any,
               pos: str = "upper left",
               offset: Tuple[float, float] = (0.02, 0.02),
               form: str = "a",
               start: int = 1,
               fstr: str = "{}",
               **kwargs: Any
               ) -> None:
    """
    Adds consecutive labels to the corners of a series of |Axes|. Useful for
    creating numbered plots.

    Parameters
    ----------
    axs : sequence of |Axes|
    pos : str, default {"upper left"}
        The location to place the text at. Right now the only option is the
        upper left corner.
    offset : (float, float), default (0.02, 0.02)
        The horizontal and vertical offsets from the edges of the Axes, in Axes
        coordinates. In other words, the default offset places 2% of space
        between the edge of the Axes and the text.
    form : str from {"a", "A", "i", "I", "1"}, default "a"
        The numbers, or characters, to use for numbering. The alphabet work up
        to 26 times (i.e. 'z'). Roman numerals work up to 26 (i.e. 'xxvi'). If
        you need more than that, you're probably doing your plots wrongly.
    start : int, default 1
        The number to start from. Values below 1 are ignored.
    fstr : str, default "{}"
        A format string with a single field. The actual string placed on the
        axes will be fstr.format(char), where char is the actual counter. Thus,
        if you want your graphs to be labelled (a), (b), ... set fstr to be
        "({})", for example.
    **kwargs : dict
        Optional parameters which are passed to ax.text(). If there is a clash,
        the parameters passed here will override choices made using the other
        keyword arguments. For example, if you want your text in the middle of
        the spectrum, you can call `label_axes(..., x=0.5, y=0.5)`.
    """
    # Convert whatever sequence we throw at it
    if isinstance(axs, matplotlib.axes.Axes):
        axs = [axs]
    axs_flat = np.array(axs).flat
    # Figure out text placement. We need to use kwargs.pop() to make sure that
    # we don't pass the same keyword argument twice to ax.text() later on.
    if pos == "upper left":
        x = kwargs.pop("x", 0 + offset[0])
        y = kwargs.pop("y", 1 - offset[1])
        ha = kwargs.pop("horizontalalignment", "left")
        va = kwargs.pop("verticalalignment", "top")
    else:
        raise ValueError(f"invalid value '{pos}' provided for pos")
    # Figure out other parameters. This is a bit odd, but we need to do this to
    # (1) avoid passing the same argument twice to ax.text(), as before, but
    # also (2) avoid popping the value on the first Axes and then not having it
    # for subsequent ones. This is not an issue for x, y, ha, and va because
    # those are constants for all Axes, whereas s and transform are in general
    # not.
    kwargs_s = kwargs.pop("s", None)
    kwargs_trfm = kwargs.pop("trfm", None)
    # Prime the character generator
    gen = label_generator(form)
    while start > 1:
        next(gen)
        start = start - 1
    # Do the plotting
    for ax in axs_flat:
        s = kwargs_s or fstr.format(next(gen))
        trfm = kwargs_trfm or ax.transAxes
        ax.text(x=x, y=y, s=s, transform=trfm,
                horizontalalignment=ha, verticalalignment=va, **kwargs)
