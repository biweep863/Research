"""Live speaker radar for the ReSpeaker (or a recorded / simulated 6-ch WAV).

    python -m src.experiments.live_radar                       # live, auto-find ReSpeaker
    python -m src.experiments.live_radar --device 3            # explicit sounddevice index
    python -m src.experiments.live_radar --wav data/demo_moving_speaker.wav
    python -m src.experiments.live_radar --wav x.wav --speed 0 --no-plot --log out.csv

Radar: robot at the centre, its front at the top. Big dot = tracked talker
(radius = estimated distance, opacity = confidence; hollow on the outer ring
while no distance is available). Small x = raw per-frame DOA. Grey outline =
SRP-PHAT map (where the array "hears" energy). Dashed ray = firmware DOAANGLE
(when the USB control interface is reachable) for comparison.
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import deque
from pathlib import Path

import numpy as np
import yaml

from ..analysis.distance import DistanceModel
from ..analysis.geometry import array_to_robot_deg
from ..analysis.localizer import Localizer
from ..data.respeaker_io import FirmwarePoller, LiveSource, WavSource, find_device, list_input_devices, select_mics


def load_model(cfg, override=None, rt60_known: bool = False) -> DistanceModel:
    """Priority: --model > hardware calibration > sim prior (RT60 variant if the RT60 is known)."""
    d = cfg["distance"]
    rt60_known = rt60_known or d.get("room_rt60_s") is not None
    sim = [d.get("model_rt60_path"), d.get("model_path")] if rt60_known else [d.get("model_path")]
    for p in [override, d.get("hw_model_path"), *sim]:
        if p and Path(p).exists():
            m = DistanceModel.from_json(p)
            print(f"distance model: {p} ({m.source})")
            return m
    print("distance model: untrained inverse-square prior (run `make run` or `make calibrate`)")
    return DistanceModel(ref_level_db=d["ref_level_db"], max_range=d["max_range_m"])


class Radar:
    def __init__(self, cfg):
        import matplotlib.pyplot as plt

        self.plt = plt
        r = cfg["radar"]
        self.rmax = float(r["max_range_m"])
        self.trail = deque(maxlen=int(r["trail"]))
        plt.ion()
        self.fig = plt.figure(figsize=(7.5, 7.5))
        self.fig.canvas.manager.set_window_title("ReSpeaker speaker radar")
        ax = self.ax = self.fig.add_subplot(projection="polar")
        ax.set_theta_zero_location(r["zero_location"])
        ax.set_theta_direction(r["direction"])
        ax.set_rlim(0, self.rmax)
        ax.set_rticks(np.arange(1, self.rmax + 0.01, 1.0))
        ax.set_yticklabels([f"{k:.0f} m" for k in np.arange(1, self.rmax + 0.01, 1.0)], fontsize=8)
        ax.grid(alpha=0.35)
        ax.plot([0], [0], marker=(3, 0, 0), ms=14, color="k")  # robot, front up
        th = np.deg2rad(np.arange(0, 361, 1.0))
        (self.srp_line,) = ax.plot(th, np.full_like(th, np.nan), color="0.55", lw=1.0)
        self.trail_sc = ax.scatter([], [], s=18, c=[], cmap="Blues", vmin=0, vmax=1)
        self.meas_sc = ax.scatter([], [], marker="x", s=30, color="tab:orange", alpha=0.8)
        self.blip = ax.scatter([], [], s=420, color="tab:red", edgecolors="darkred", linewidths=2, zorder=5)
        (self.fw_line,) = ax.plot([], [], ls="--", color="tab:green", lw=1.5, label="firmware DOA")
        self.text = self.fig.text(0.02, 0.01, "", family="monospace", fontsize=9, va="bottom")
        self.title = ax.set_title("waiting for audio...", pad=18)
        self.fig.subplots_adjust(top=0.9, bottom=0.16, left=0.08, right=0.92)
        self.closed = False
        self.fig.canvas.mpl_connect("close_event", lambda e: setattr(self, "closed", True))

    def update(self, est, srp, cfg, fw_doa, extra: str):
        a = cfg["audio"]
        to_robot = lambda az: array_to_robot_deg(az, a["azimuth_offset_deg"], a["mirror"])  # noqa: E731
        if srp is not None:
            grid = np.arange(0, 360, a["grid_step_deg"])
            th = np.deg2rad(to_robot(grid))
            order = np.argsort(th)
            rr = self.rmax * (0.55 + 0.45 * np.clip(srp, 0, 1))
            th, rr = th[order], rr[order]
            self.srp_line.set_data(np.r_[th, th[:1] + 2 * np.pi], np.r_[rr, rr[:1]])
        if est is not None and np.isfinite(est.az_deg):
            self.meas_sc.set_offsets([[np.deg2rad(to_robot(est.az_deg)), self.rmax * 0.97]])
        if est is not None and est.track_active:
            th = np.deg2rad(to_robot(est.track_az_deg))
            has_r = np.isfinite(est.track_range_m)
            r = min(est.track_range_m, self.rmax * 0.95) if has_r else self.rmax * 0.9
            self.blip.set_offsets([[th, r]])
            self.blip.set_alpha(float(np.clip(0.25 + est.track_conf, 0.25, 1.0)))
            self.blip.set_facecolor("tab:red" if has_r else "none")
            self.trail.append((th, r))
        else:
            self.blip.set_offsets(np.empty((0, 2)))
        if self.trail:
            pts = np.array(self.trail)
            self.trail_sc.set_offsets(pts)
            self.trail_sc.set_array(np.linspace(0.1, 0.8, len(pts)))
        if fw_doa is not None:
            th = np.deg2rad(to_robot(fw_doa))  # same frame mapping; verify on HW
            self.fw_line.set_data([th, th], [0, self.rmax])
        state = "SPEECH" if (est and est.is_speech) else "silence"
        self.title.set_text(f"{state}" + (f"   F0 {est.f0_hz:5.0f} Hz" if est and est.is_speech else "")
                            + (f"   {est.range_band.upper()}" if est and est.range_band else ""))
        self.text.set_text(extra)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()


def _fmt(est, loc, fw):
    if est is None:
        return ""
    f = loc.last_features or {}
    lines = [
        f"t {est.t:7.2f}s  rms {20 * np.log10(est.rms + 1e-12):6.1f} dBFS  gate {20 * np.log10(est.threshold + 1e-12):6.1f}",
        f"track az {est.track_az_deg:6.1f}  range {est.track_range_m:4.2f} m  conf {est.track_conf:.2f}"
        + (f"  [{est.range_band.upper()}]" if est.range_band else ""),
        "cdr {:5.1f} dB  level {:6.1f}  speech {:6.1f}  snr {:5.1f}".format(
            f.get("cdr_db", np.nan), f.get("level_db", np.nan), f.get("speech_db", np.nan), f.get("snr_db", np.nan)),
    ]
    if loc.rel is not None and (ref := loc.rel.reference()) is not None:
        lines.append(f"relative ref {ref[0]:6.1f} dB +-{ref[1]:.1f}  ({len(loc.rel.hist)} utterances learned)")
    if fw is not None and fw.doa is not None:
        lines.append(f"firmware DOA {fw.doa:3d}  VAD {fw.vad}  RT60 {fw.rt60 if fw.rt60 is None else round(fw.rt60, 2)}")
    return "\n".join(lines)


LOG_FIELDS = ["t", "is_speech", "f0_hz", "rms", "az_meas", "doa_conf", "range_meas", "track_az_robot",
              "track_range_m", "track_conf", "cdr_db", "level_db", "snr_db", "tilt_db", "fw_doa",
              "speech_db", "range_band", "model_range_m"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/local_small.yaml")
    ap.add_argument("--wav", help="replay a 6-ch (or 4-ch raw) WAV instead of the live device")
    ap.add_argument("--speed", type=float, default=1.0, help="replay speed; 0 = as fast as possible")
    ap.add_argument("--device", help="sounddevice index or name (default: auto-find ReSpeaker)")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--model", help="distance model JSON (default: hw calibration > sim prior)")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--no-firmware", action="store_true", help="do not poll firmware DOA over USB")
    ap.add_argument("--log", help="write per-frame CSV")
    ap.add_argument("--save-png", help="save the final radar frame (wav mode)")
    ap.add_argument("--duration", type=float, default=0, help="stop after N seconds (0 = forever)")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    a = cfg["audio"]

    if args.list_devices:
        for i, n, ch in list_input_devices():
            print(f"{i:3d}  {ch:2d} ch  {n}")
        return

    loc = Localizer(cfg, load_model(cfg, args.model))
    if args.wav:
        src = WavSource(args.wav, args.speed)
        fw = None
    else:
        dev = args.device
        dev = int(dev) if dev and dev.isdigit() else dev
        if dev is None:
            dev = find_device(a["device_name"], a["device_channels"])
            if dev is None:
                raise SystemExit(f"No input device matching '{a['device_name']}' with "
                                 f"{a['device_channels']} channels. Use --list-devices / --device. "
                                 "Is the 6-channel firmware flashed?")
        src = LiveSource(dev, a["device_channels"], a["fs"], a["hop"] * 2)
        fw = None if args.no_firmware else FirmwarePoller()
        if fw is not None and fw.error:
            print(f"firmware DOA unavailable: {fw.error}")

    radar = None if args.no_plot else Radar(cfg)
    log_f = open(args.log, "w", newline="") if args.log else None
    writer = csv.writer(log_f) if log_f else None
    if writer:
        writer.writerow(LOG_FIELDS)

    rt60_switched = False
    fps = cfg["radar"]["fps"]
    last_draw = last_print = 0.0
    last_est = None
    t_start = time.monotonic()
    try:
        with src:
            while True:
                block = src.read()
                if block is not None:
                    mics = select_mics(block, a["mic_channels"])
                    for est in loc.process(mics):
                        last_est = est
                        if writer:
                            f = est.features or {}
                            writer.writerow([
                                f"{est.t:.3f}", int(est.is_speech), f"{est.f0_hz:.1f}", f"{est.rms:.6f}",
                                f"{est.az_deg:.1f}", f"{est.doa_conf:.3f}", f"{est.range_m:.2f}",
                                f"{array_to_robot_deg(est.track_az_deg, a['azimuth_offset_deg'], a['mirror']):.1f}",
                                f"{est.track_range_m:.2f}", f"{est.track_conf:.3f}",
                                *(f"{f.get(k, np.nan):.2f}" for k in ("cdr_db", "level_db", "snr_db", "tilt_db")),
                                "" if fw is None or fw.doa is None else fw.doa,
                                f"{f.get('speech_db', np.nan):.2f}", est.range_band or "",
                                f"{f.get('model_range_m', np.nan):.2f}",
                            ])
                now = time.monotonic()
                fw_doa = None if fw is None else fw.doa
                if (not rt60_switched and fw is not None and fw.rt60 and loc.room_rt60 is None
                        and not args.model):
                    # firmware RT60 estimate available: use it as the room covariate
                    loc.room_rt60 = fw.rt60
                    loc.model = load_model(cfg, None, rt60_known=True)
                    print(f"using firmware RT60 {fw.rt60:.2f} s as room covariate")
                    rt60_switched = True
                if radar and now - last_draw > 1.0 / fps:
                    radar.update(last_est, loc.last_srp, cfg, fw_doa, _fmt(last_est, loc, fw))
                    last_draw = now
                    if radar.closed:
                        break
                elif not radar and last_est is not None and now - last_print > 0.25:
                    e = last_est
                    print(f"t={e.t:6.2f} {'SPEECH ' if e.is_speech else 'silence'} f0={e.f0_hz:5.0f} "
                          f"az={array_to_robot_deg(e.track_az_deg, a['azimuth_offset_deg'], a['mirror']):6.1f} "
                          f"r={e.track_range_m:4.2f} {e.range_band or '':4s} conf={e.track_conf:.2f}"
                          + (f" fw={fw_doa}" if fw_doa is not None else ""))
                    last_print = now
                if isinstance(src, WavSource) and src.done:
                    break
                if args.duration and now - t_start > args.duration:
                    break
                if block is None:
                    if radar:
                        radar.plt.pause(0.005)
                    else:
                        time.sleep(0.005)
    except KeyboardInterrupt:
        pass
    finally:
        if fw is not None:
            fw.stop()
        if log_f:
            log_f.close()
            print(f"wrote {args.log}")
    if radar and args.save_png:
        radar.update(last_est, loc.last_srp, cfg, None, _fmt(last_est, loc, fw))
        radar.fig.savefig(args.save_png, dpi=120)
        print(f"wrote {args.save_png}")


if __name__ == "__main__":
    main()
