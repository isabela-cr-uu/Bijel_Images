import math
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

COLOR_CHOICES = [
    "#FE0056",
    "#FF8000",
]

RADIANS_PER_45_DEGREES = math.pi / 4

class PolygonDrawer:

    def __init__(self, root):

        self.root = root
        self.root.title("Polygon Drawer")

        self.canvas = tk.Canvas(root, bg="white")
        self.canvas.pack(fill="both", expand=True)

        self.image = None
        self.tk_image = None

        self.points = []
        self.polygons = []
        self.colors = []

        self.current_color = COLOR_CHOICES[0]

        self.constrain_angles = False

        # mouse events
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Motion>", self.on_mouse_move)

        # keyboard
        self.root.bind("<Return>", lambda e: self.close_polygon())
        self.root.bind("<Control-z>", lambda e: self.undo())

        self.root.bind("<KeyPress-Shift_L>", self.shift_down)
        self.root.bind("<KeyRelease-Shift_L>", self.shift_up)

        self.create_toolbar()

    def create_toolbar(self):

        frame = tk.Frame(self.root)
        frame.pack(fill="x")

        tk.Button(
            frame,
            text="Load Image",
            command=self.load_image
        ).pack(side="left")

        tk.Button(
            frame,
            text="Undo",
            command=self.undo
        ).pack(side="left")

        tk.Button(
            frame,
            text="Clear All",
            command=self.clear_all
        ).pack(side="left")

    def load_image(self):

        path = filedialog.askopenfilename(
            filetypes=[
                (
                    "Images",
                    "*.png *.jpg *.jpeg *.webp *.tif *.tiff *.lif *.liff"
                )
            ]
        )

        if not path:
            return

        self.image = Image.open(path)

        self.tk_image = ImageTk.PhotoImage(self.image)

        self.canvas.config(
            width=self.image.width,
            height=self.image.height
        )

        self.redraw()

    def redraw(self):

        self.canvas.delete("all")

        if self.tk_image:
            self.canvas.create_image(
                0,
                0,
                anchor="nw",
                image=self.tk_image
            )

        for i, polygon in enumerate(self.polygons):

            color = self.colors[i]

            if len(polygon) >= 2:

                flat = [coord for point in polygon for coord in point]

                self.canvas.create_polygon(
                    flat,
                    outline=color,
                    fill="",
                    width=3
                )

            for x, y in polygon:
                self.draw_node(x, y)

        if len(self.points) > 0:

            for i in range(len(self.points) - 1):

                x1, y1 = self.points[i]
                x2, y2 = self.points[i + 1]

                self.canvas.create_line(
                    x1,
                    y1,
                    x2,
                    y2,
                    fill=self.current_color,
                    width=3
                )

            for x, y in self.points:
                self.draw_node(x, y)

    def draw_node(self, x, y):

        r = 4

        self.canvas.create_oval(
            x - r,
            y - r,
            x + r,
            y + r,
            fill="white",
            outline="black"
        )

    def on_click(self, event):

        x = event.x
        y = event.y

        # angle snapping
        if self.constrain_angles and self.points:

            x, y = self.snap_angle(x, y)

        # close polygon if clicked near first point
        if len(self.points) > 2:

            px, py = self.points[0]

            dist = math.sqrt((x - px) ** 2 + (y - py) ** 2)

            if dist <= 10:
                self.close_polygon()
                return

        self.points.append([x, y])

        self.redraw()

    def on_mouse_move(self, event):

        self.redraw()

        if len(self.points) == 0:
            return

        x = event.x
        y = event.y

        if self.constrain_angles:
            x, y = self.snap_angle(x, y)

        x1, y1 = self.points[-1]

        self.canvas.create_line(
            x1,
            y1,
            x,
            y,
            fill=self.current_color,
            width=2
        )

    def snap_angle(self, x, y):

        last_x, last_y = self.points[-1]

        dx = x - last_x
        dy = y - last_y

        angle = math.atan2(dy, dx)

        length = math.sqrt(dx * dx + dy * dy)

        snapped = round(
            angle / RADIANS_PER_45_DEGREES
        ) * RADIANS_PER_45_DEGREES

        new_x = last_x + length * math.cos(snapped)
        new_y = last_y + length * math.sin(snapped)

        return round(new_x), round(new_y)

    def close_polygon(self):

        if len(self.points) < 3:
            return

        self.polygons.append(self.points.copy())

        self.colors.append(self.current_color)

        self.current_color = COLOR_CHOICES[
            len(self.colors) % len(COLOR_CHOICES)
        ]

        self.points = []

        self.redraw()

    def undo(self):

        if len(self.points) > 0:

            self.points.pop()

            self.redraw()

    def clear_all(self):

        self.points = []

        self.polygons = []

        self.colors = []

        self.current_color = COLOR_CHOICES[0]

        self.redraw()

    def shift_down(self, event):

        self.constrain_angles = True

    def shift_up(self, event):

        self.constrain_angles = False

if __name__ == "__main__":

    root = tk.Tk()

    app = PolygonDrawer(root)

    root.mainloop()