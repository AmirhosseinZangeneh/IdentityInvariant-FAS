"""Local opaque single-image Tk reviewer; no annotation targets or analysis imports."""
from __future__ import annotations

import time
import tkinter as tk
from tkinter import messagebox, simpledialog

from PIL import Image, ImageTk

from .msu_pilot import GRID_TRIALS, SIDES, Presenter, display_to_image


def display_metadata(window, disclosure):
    return dict(tk_scaling=float(window.tk.call("tk", "scaling")),
                dpi=float(window.winfo_fpixels("1i")),
                screen=[window.winfo_screenwidth(), window.winfo_screenheight()],
                os_scaling_disclosure=disclosure)


class ReviewerWindow:
    def __init__(self, window, presenter, disclosure):
        self.window, self.presenter = window, presenter
        self.disclosure = disclosure
        self.display = display_metadata(window, disclosure)
        self.started = time.monotonic()
        self.ended = False
        self.trials, self.count, self.payload = [], 0, None
        self.boundary, self.drag = [20, 20], None
        self.centers = {}
        window.title("Exploratory pupil localization — masked review")
        window.protocol("WM_DELETE_WINDOW", self.stop_dialog)
        self.status = tk.StringVar()
        tk.Label(window, textvariable=self.status, wraplength=900).pack()
        self.canvas = tk.Canvas(window, width=920, height=560, background="#444444",
                                highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>", self.click)
        self.canvas.bind("<ButtonPress-3>", self.pan_start)
        self.canvas.bind("<B3-Motion>", self.pan_move)
        bar = tk.Frame(window)
        bar.pack()
        self.zoom = tk.IntVar(value=1)
        self.zoom_buttons = []
        for z in (1, 2, 4):
            button = tk.Radiobutton(bar, text=f"{z}x", variable=self.zoom, value=z,
                                    command=self.redraw_zoom)
            button.pack(side="left")
            self.zoom_buttons.append(button)
        tk.Label(bar, text="Right-drag to pan. Mark the visible pupil center; no fallback landmark.").pack(side="left")
        self.side = tk.StringVar(value=SIDES[0])
        self.fields = {}
        for side in SIDES:
            row = tk.Frame(window)
            row.pack(fill="x")
            tk.Radiobutton(row, text=side, variable=self.side, value=side).pack(side="left")
            missing, radius, reason = tk.BooleanVar(), tk.StringVar(value="0.5"), tk.StringVar()
            tk.Checkbutton(row, text="Unmeasurable", variable=missing).pack(side="left")
            tk.Label(row, text="Radius (original pixels)").pack(side="left")
            tk.Entry(row, textvariable=radius, width=8).pack(side="left")
            tk.Label(row, text="Reason / uncertainty explanation").pack(side="left")
            tk.Entry(row, textvariable=reason, width=45).pack(side="left")
            self.fields[side] = (missing, radius, reason)
        actions = tk.Frame(window)
        actions.pack()
        self.save = tk.Button(actions, text="Save both eyes and advance", command=self.submit, state="disabled")
        self.save.pack(side="left")
        tk.Button(actions, text="Record recognition / disclosure", command=self.note).pack(side="left")
        tk.Button(actions, text="Stop: target exposure, fatigue, or problem", command=self.stop_dialog).pack(side="left")
        self.grid_image()
        window.after(1000, self.tick)

    def guard_display(self):
        if display_metadata(self.window, self.disclosure) != self.display:
            raise ValueError("Display scaling changed")

    def tick(self):
        if self.ended:
            return
        if time.monotonic() - self.started >= 3600:
            self.stop("60-minute session limit reached")
            return
        try:
            self.guard_display()
        except ValueError:
            self.stop("Display scaling changed; synthetic validation no longer applicable")
            return
        self.window.after(1000, self.tick)

    def grid_image(self):
        z, boundary, target = GRID_TRIALS[len(self.trials)]
        self.zoom.set(z)
        self.boundary = list(boundary)
        image = Image.new("RGB", (64, 64), "white")
        for y in range(64):
            for x in range(64):
                if x % 8 == 0 or y % 8 == 0:
                    image.putpixel((x, y), (180, 180, 180))
        # Synthetic-only locator surrounds the single green pixel; never used on real images.
        for y in range(target[1]-2, target[1]+3):
            for x in range(target[0]-2, target[0]+3):
                image.putpixel((x, y), (255, 100, 100))
        image.putpixel(target, (0, 180, 0))
        self.image = image
        self.status.set(f"Synthetic coordinate check {len(self.trials)+1}/6: click the green pixel center. "
                        "A failed click stops this session; no candidate is shown.")
        for button in self.zoom_buttons:
            button.configure(state="disabled")
        self.render()

    def render(self):
        self.canvas.delete("all")
        z = self.zoom.get()
        self.photo = ImageTk.PhotoImage(self.image.resize(
            (self.image.width*z, self.image.height*z), Image.Resampling.NEAREST))
        self.canvas.create_image(*self.boundary, image=self.photo, anchor="nw", tags="image")

    def redraw_zoom(self):
        if self.payload is not None and not self.ended:
            self.boundary = [20, 20]
            self.render()

    def pan_start(self, event):
        if self.payload is not None:
            self.drag = (event.x, event.y)

    def pan_move(self, event):
        if self.drag is not None and self.payload is not None:
            dx, dy = event.x-self.drag[0], event.y-self.drag[1]
            self.boundary = [self.boundary[0]+dx, self.boundary[1]+dy]
            self.canvas.move("image", dx, dy)
            self.drag = (event.x, event.y)

    def click(self, event):
        if self.ended or (len(self.trials) == 6 and self.payload is None):
            return
        try:
            self.guard_display()
            click = [event.x, event.y]
            if len(self.trials) < 6:
                z, b, target = GRID_TRIALS[len(self.trials)]
                xy = display_to_image(click, b, z)
                if any(abs(a-c) > 0.5 for a, c in zip(xy, target)):
                    self.stop("Synthetic coordinate validation failed")
                    return
                self.trials.append(dict(zoom=z, boundary=list(b), target=list(target), click=click,
                                        converted=xy, viewport=[self.canvas.winfo_width(), self.canvas.winfo_height()]))
                if len(self.trials) < 6:
                    self.grid_image()
                else:
                    self.presenter.grid(self.trials, self.display)
                    self.show_current()
                return
            xy = display_to_image(click, self.boundary, self.zoom.get())
            if not (0 <= xy[0] <= self.image.width-1 and 0 <= xy[1] <= self.image.height-1):
                messagebox.showinfo("Outside image", "Click a pupil inside the image.")
                return
            self.centers[self.side.get()] = dict(x=xy[0], y=xy[1], conversion=dict(
                click=click, boundary=list(self.boundary), zoom=self.zoom.get(),
                viewport=[self.canvas.winfo_width(), self.canvas.winfo_height()], display=self.display))
            self.status.set(self.payload["opaque_id"] + ": " + self.side.get() + " center selected.")
        except (ValueError, OSError, KeyError, TypeError):
            self.stop("Coordinate or record validation failed")

    def show_current(self):
        self.payload = self.presenter.current()
        self.image = Image.fromarray(self.payload["pixels"][:, :, ::-1])
        self.centers = {}
        self.side.set(SIDES[0])
        for missing, radius, reason in self.fields.values():
            missing.set(False)
            radius.set("0.5")
            reason.set("")
        self.zoom.set(1)
        self.boundary = [20, 20]
        self.status.set(f"{self.payload['opaque_id']} — image {self.count+1}/16. "
                        "Choose each eye, click its pupil, and state uncertainty or mark unmeasurable.")
        for button in self.zoom_buttons:
            button.configure(state="normal")
        self.save.configure(state="normal")
        self.render()

    def submit(self):
        if self.ended or self.payload is None:
            return
        try:
            self.guard_display()
        except ValueError:
            self.stop("Display scaling changed; synthetic validation no longer applicable")
            return
        try:
            eyes = {}
            for side, (missing, radius, reason) in self.fields.items():
                if missing.get():
                    eyes[side] = dict(measurable=False, x=None, y=None, radius=None,
                                      reason=reason.get(), conversion=None)
                else:
                    if side not in self.centers:
                        messagebox.showinfo("Missing mark", "Mark both eyes or state why an eye is unmeasurable.")
                        return
                    eyes[side] = dict(measurable=True, **self.centers[side],
                                      radius=float(radius.get()), reason=reason.get())
            # Validate editable fields before append; an invalid form does not lose a session.
            from .msu_pilot import validate_eyes
            validate_eyes(eyes, [self.image.height, self.image.width, 3])
        except (ValueError, TypeError):
            messagebox.showinfo("Correct marks/form", "Image-left must be left of image-right. Correct the marks; they are never swapped. Use finite radii of at least 0.5 pixels and explain larger radii or unmeasurable eyes.")
            return
        try:
            self.presenter.submit(self.payload["opaque_id"], eyes)
            self.count += 1
            self.canvas.delete("all")
            self.payload = None
            if self.count == 16:
                self.presenter.finish()
                self.ended = True
                self.save.configure(state="disabled")
                self.status.set("Session locked. No scores or candidate mappings are available in this interface.")
            elif self.count == 8:
                self.save.configure(state="disabled")
                self.status.set("Take a break. Resume when ready; the 60-minute session limit includes this break.")
                self.resume = tk.Button(self.window, text="Resume after break", command=self.resume_break)
                self.resume.pack()
            else:
                self.show_current()
        except (ValueError, OSError, KeyError, TypeError):
            self.stop("Record or asset verification failed")

    def resume_break(self):
        if self.ended or self.payload is not None or self.count != 8:
            return
        try:
            self.presenter.resume_break()
            self.resume.destroy()
            self.show_current()
        except (ValueError, OSError, KeyError, TypeError):
            self.stop("Break or asset verification failed")

    def note(self):
        if self.ended:
            return
        text = simpledialog.askstring("Disclosure", "Record recognition or other exposure.")
        if text and text.strip():
            if messagebox.askyesno("Exposure check", "Were hidden annotation targets exposed during marking? Yes invalidates this session."):
                self.stop("Invalidating target exposure: " + text)
                return
            try:
                self.presenter.note(text)
            except (ValueError, OSError):
                self.stop("Could not record disclosure")

    def stop(self, reason):
        if not self.ended:
            try:
                self.presenter.stop(reason)
            except (ValueError, OSError):
                pass  # Partial records remain; absence of a valid lock blocks all analysis.
        self.ended = True
        self.canvas.delete("all")
        self.payload = None
        self.save.configure(state="disabled")
        self.status.set("STOP. Session incomplete/invalid; coordinator must review. No automatic restart.")

    def stop_dialog(self):
        if self.ended:
            self.window.destroy()
            return
        reason = simpledialog.askstring("Stop session", "Reason (including any target exposure, fatigue, or software problem):")
        if reason and reason.strip():
            self.stop(reason)


def enable_windows_dpi(user32):
    """Fail before Tk/session creation unless the requested coordinate system is set."""
    import ctypes
    if not user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
        raise ValueError("Cannot establish Windows per-monitor DPI awareness")


def run(root, session):
    # Use one consistent per-monitor coordinate system on Windows before creating Tk.
    import sys
    if sys.platform == "win32":
        import ctypes
        enable_windows_dpi(ctypes.windll.user32)
    window = tk.Tk()
    window.withdraw()
    if not messagebox.askyesno("Reviewer identity", "Are you the actual reviewer named in the frozen plan: "
                              + session.plan["reviewer"] + "? Continue only with separate execution authorization."):
        window.destroy()
        return
    disclosure = simpledialog.askstring("Display configuration", "Record Windows display scaling (e.g. 100%), monitor and any remote desktop scaling:")
    if not disclosure or not disclosure.strip():
        window.destroy()
        return
    presenter = Presenter(root, session)
    ui = None
    try:
        presenter.start()
        ui = ReviewerWindow(window, presenter, disclosure)
        # Tk's default exception handler prints tracebacks. Keep provenance out of reviewer output.
        def callback_failed(*_):
            ui.stop("Unexpected interface failure; session invalid")
        window.report_callback_exception = callback_failed
        window.deiconify()
        window.mainloop()
    finally:
        if ui is not None and not ui.ended:
            ui.stop("Interface terminated before valid completion")
        try:
            window.destroy()
        except tk.TclError:
            pass
