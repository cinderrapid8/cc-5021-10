"""Matplotlib teaching UI: editable targets, skeleton inspection and export."""

from __future__ import annotations

from pathlib import Path
import numpy as np

from .skeleton import HandSkeleton, FINGER_NAMES
from .flute import Flute
from .timeline import default_timeline
from .animation import PoseSampler
from .render import render_pose
from .export import export_joint_curves, export_contact_markers


class FingeringApp:
    def __init__(self, out_dir: str = "output"):
        import matplotlib
        try:
            import tkinter  # noqa: F401
            matplotlib.use("TkAgg", force=True)
        except Exception:
            matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider, Button, RadioButtons

        self.plt = plt
        self.out_dir = Path(out_dir)
        self.sk = HandSkeleton()
        self.flute = Flute()
        self.timeline = default_timeline()
        self.sampler = PoseSampler(self.sk, self.flute, self.timeline)
        self.selected_finger = 0
        self.playing = False
        self._timer = None

        fig = plt.figure(figsize=(13, 7.5))
        gs = fig.add_gridspec(
            2, 2, height_ratios=[4, 1], width_ratios=[2, 1], hspace=0.25, wspace=0.18
        )
        self.ax_side = fig.add_subplot(gs[0, 0])
        self.ax_top = fig.add_subplot(gs[0, 1])
        ax_time = fig.add_axes([0.08, 0.10, 0.55, 0.03])
        ax_trans = fig.add_axes([0.08, 0.05, 0.55, 0.03])
        self.slider_t = Slider(
            ax_time, "time (s)", 0.0, self.timeline.duration, valinit=0.0
        )
        self.slider_trans = Slider(
            ax_trans, "transition (s)", 0.0, 0.6, valinit=self.timeline.transition
        )
        self.slider_t.on_changed(self._on_time)
        self.slider_trans.on_changed(self._on_transition)

        self._add_buttons(fig)
        self.fig = fig
        self._redraw()
        fig.canvas.mpl_connect("pick_event", self._on_pick)

    # ---------------------------------------------------------------- widgets
    def _add_buttons(self, fig):
        from matplotlib.widgets import Button, RadioButtons

        ax_play = fig.add_axes([0.72, 0.10, 0.08, 0.04])
        ax_reset = fig.add_axes([0.82, 0.10, 0.08, 0.04])
        ax_export = fig.add_axes([0.72, 0.05, 0.18, 0.04])
        Button(ax_play, "play / pause").on_clicked(self._toggle_play)
        Button(ax_reset, "reset target").on_clicked(self._reset_target)
        Button(ax_export, "export curves + markers").on_clicked(self._export)
        ax_radio = fig.add_axes([0.92, 0.35, 0.07, 0.25])
        self.radio = RadioButtons(ax_radio, FINGER_NAMES, active=0)
        self.radio.on_clicked(self._select_finger)

    # ----------------------------------------------------------------- events
    def _on_time(self, _):
        self._redraw()

    def _on_transition(self, val):
        self.timeline.set_transition(float(val))
        self._redraw()

    def _select_finger(self, label):
        self.selected_finger = FINGER_NAMES.index(label)

    def _toggle_play(self, _):
        self.playing = not self.playing
        if self.playing:
            self._timer = self.fig.canvas.new_timer(interval=50)
            self._timer.add_callback(self._tick)
            self._timer.start()
        elif self._timer is not None:
            self._timer.stop()

    def _tick(self):
        t = self.slider_t.val + 0.03
        if t >= self.timeline.duration:
            t = 0.0
        self.slider_t.set_val(t)

    def _on_pick(self, event):
        # Drag the picked finger's target by clicking its target ring.
        if event.mouseevent.inaxes not in (self.ax_side, self.ax_top):
            return
        me = event.mouseevent
        artist = event.artist
        f = getattr(artist, "finger", self.selected_finger)
        self.selected_finger = f
        cur = np.array(
            self.flute.overrides.get(f, self.flute.hole_position(f + 1)), dtype=float
        )
        if me.inaxes is self.ax_side:
            cur[1], cur[2] = me.xdata, me.ydata
        else:
            cur[1], cur[0] = me.xdata, me.ydata
        self.flute.overrides[f] = cur
        self.radio.set_active(f)
        self._redraw()

    def _reset_target(self, _):
        self.flute.overrides.pop(self.selected_finger, None)
        self._redraw()

    def _export(self, _):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        jc = export_joint_curves(self.sampler, self.out_dir / "joint_curves.csv")
        cm = export_contact_markers(self.sampler, self.out_dir / "contact_markers.csv")
        print(f"exported {jc} and {cm}")

    # ------------------------------------------------------------------ render
    def _redraw(self):
        pose = self.sampler.pose_at(float(self.slider_t.val))
        render_pose(self.ax_side, self.ax_top, self.sk, self.flute, pose)
        self.fig.canvas.draw_idle()

    def run(self):
        self.plt.show()


def main():
    FingeringApp().run()


if __name__ == "__main__":
    main()
