import sys, math
import sexpdata
from PySide6.QtWidgets import QApplication, QGraphicsView, QGraphicsScene, QGraphicsLineItem, QGraphicsEllipseItem, QGraphicsRectItem, QGraphicsPathItem
from PySide6.QtGui import QPen, QPainterPath
from PySide6.QtCore import Qt, QRectF

def compute_circle_from_points(A, B, C):
    (x1, y1), (x2, y2), (x3, y3) = A, B, C
    D = 2 * (x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
    if D == 0:
        return None, None  # colinear
    center_x = ((x1**2 + y1**2)*(y2-y3) + (x2**2+y2**2)*(y3-y1) + (x3**2+y3**2)*(y1-y2)) / D
    center_y = ((x1**2 + y1**2)*(x3-x2) + (x2**2+y2**2)*(x1-x3) + (x3**2+y3**2)*(x2-x1)) / D
    radius = math.hypot(x1 - center_x, y1 - center_y)
    return (center_x, center_y), radius

def angle_between(center, point):
    return math.degrees(math.atan2(point[1]-center[1], point[0]-center[0]))

def normalize_angle(angle):
    return angle % 360

def traverse(expr):
    """Yield all lists in the S-expression tree."""
    if isinstance(expr, list):
        yield expr
        for sub in expr:
            yield from traverse(sub)

def get_offset(items):
    """Return the (x, y) offset from an (at x y ...) property, if present."""
    at_item = None
    for itm in items:
        if isinstance(itm, list) and itm and isinstance(itm[0], sexpdata.Symbol) and itm[0].value() == "at":
            at_item = itm
            break
    if at_item and len(at_item) >= 3:
        try:
            return (float(at_item[1]), float(at_item[2]))
        except:
            return (0, 0)
    return (0, 0)

def parse_kicad_schematic(file_path):
    with open(file_path, 'r') as file:
        data = sexpdata.loads(file.read())

    lines = []
    circles = []
    arcs = []       # Each arc as dict with keys: center, radius, start_angle, span_angle
    rectangles = [] # Each rectangle as (x, y, w, h)

    def find_item(items, keyword):
        for itm in items:
            if isinstance(itm, list) and itm and itm[0] == sexpdata.Symbol(keyword):
                return itm
        return None

    for item in traverse(data):
        if not (isinstance(item, list) and item):
            continue
        if not isinstance(item[0], sexpdata.Symbol):
            continue

        # Check for a local offset for this element.
        offset = get_offset(item[1:])

        sym = item[0].value()
        # Handle polyline
        if sym == 'polyline':
            pts_item = find_item(item[1:], 'pts')
            if pts_item:
                points = []
                for pt in pts_item[1:]:
                    if isinstance(pt, list) and pt and pt[0] == sexpdata.Symbol('xy'):
                        x = float(pt[1]) + offset[0]
                        y = float(pt[2]) + offset[1]
                        points.append((x, y))
                for i in range(len(points) - 1):
                    lines.append((*points[i], *points[i + 1]))
        # Handle circle
        elif sym == 'circle':
            center_item = find_item(item[1:], 'center')
            radius_item = find_item(item[1:], 'radius')
            if center_item and radius_item:
                x = float(center_item[1]) + offset[0]
                y = float(center_item[2]) + offset[1]
                r = float(radius_item[1])
                circles.append((x, y, r))
        # Handle arc
        elif sym == 'arc':
            start_item = find_item(item[1:], 'start')
            mid_item = find_item(item[1:], 'mid')
            end_item = find_item(item[1:], 'end')
            if start_item and mid_item and end_item:
                A = (float(start_item[1]) + offset[0], float(start_item[2]) + offset[1])
                B = (float(mid_item[1]) + offset[0], float(mid_item[2]) + offset[1])
                C = (float(end_item[1]) + offset[0], float(end_item[2]) + offset[1])
                center, radius = compute_circle_from_points(A, B, C)
                if center is None:
                    continue  # skip colinear arcs
                start_angle = normalize_angle(angle_between(center, A))
                end_angle = normalize_angle(angle_between(center, C))
                mid_angle = normalize_angle(angle_between(center, B))
                span_angle = (end_angle - start_angle) % 360
                if not (start_angle <= mid_angle <= start_angle + span_angle or
                        (start_angle + span_angle > 360 and mid_angle < (start_angle + span_angle) % 360)):
                    span_angle = span_angle - 360
                arcs.append({
                    'center': center,
                    'radius': radius,
                    'start_angle': start_angle,
                    'span_angle': span_angle
                })
        # Handle rectangle
        elif sym == 'rectangle':
            start_item = find_item(item[1:], 'start')
            end_item = find_item(item[1:], 'end')
            if start_item and end_item:
                x1 = float(start_item[1]) + offset[0]
                y1 = float(start_item[2]) + offset[1]
                x2 = float(end_item[1]) + offset[0]
                y2 = float(end_item[2]) + offset[1]
                x = min(x1, x2)
                y = min(y1, y2)
                w = abs(x2 - x1)
                h = abs(y2 - y1)
                rectangles.append((x, y, w, h))
    return lines, circles, arcs, rectangles

class SchematicViewer(QGraphicsView):
    def __init__(self, lines, circles, arcs, rectangles):
        super().__init__()
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.draw_lines(lines)
        #self.draw_circles(circles)
        #self.draw_arcs(arcs)
        self.draw_rectangles(rectangles)
        self.setSceneRect(self.scene.itemsBoundingRect())
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)
        self.resize(800, 600)

    def draw_lines(self, lines):
        pen = QPen(Qt.white)
        pen.setWidth(0.1)
        for x1, y1, x2, y2 in lines:
            line_item = QGraphicsLineItem(x1, -y1, x2, -y2)
            line_item.setPen(pen)
            self.scene.addItem(line_item)

    def draw_circles(self, circles):
        pen = QPen(Qt.black)
        for x, y, r in circles:
            ellipse_item = QGraphicsEllipseItem(x - r, -y - r, 2*r, 2*r)
            ellipse_item.setPen(pen)
            self.scene.addItem(ellipse_item)

    def draw_arcs(self, arcs):
        pen = QPen(Qt.black)
        for arc in arcs:
            center = arc['center']
            radius = arc['radius']
            rect = QRectF(center[0] - radius, -center[1] - radius, 2*radius, 2*radius)
            start_angle = -arc['start_angle']
            span_angle = -arc['span_angle']
            path = QPainterPath()
            path.arcTo(rect, start_angle, span_angle)
            arc_item = QGraphicsPathItem(path)
            arc_item.setPen(pen)
            self.scene.addItem(arc_item)

    def draw_rectangles(self, rectangles):
        pen = QPen(Qt.black)
        for x, y, w, h in rectangles:
            rect_item = QGraphicsRectItem(x, -y-h, w, h)
            rect_item.setPen(pen)
            self.scene.addItem(rect_item)

if __name__ == "__main__":
    file_path = "Skywalker.kicad_sch"  # Update file path if necessary
    lines, circles, arcs, rectangles = parse_kicad_schematic(file_path)
    print("Lines:", lines)
    print("Circles:", circles)
    print("Arcs:", arcs)
    print("Rectangles:", rectangles)
    app = QApplication(sys.argv)
    viewer = SchematicViewer(lines, circles, arcs, rectangles)
    viewer.setWindowTitle("KiCad Schematic Viewer")
    viewer.show()
    sys.exit(app.exec())