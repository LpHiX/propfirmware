import sys, re
from PySide6.QtWidgets import (QApplication, QGraphicsView, QGraphicsScene, 
                               QGraphicsTextItem, QGraphicsPolygonItem, QGraphicsEllipseItem)
from PySide6.QtGui import QPolygonF, QPainterPath, QColor, QPen, QBrush
from PySide6.QtCore import QPointF

# A simple S-Expression parser that ignores stray closing parens.
def tokenize(text):
    tokenPattern = r'[\(\)]|"(?:\\.|[^"])*"|[^\s\(\)]+'  # match parentheses, quoted strings, or other tokens
    tokens = re.findall(tokenPattern, text)
    return tokens

def parse_expr(tokens):
    if not tokens:
        raise ValueError("Unexpected EOF")
    token = tokens.pop(0)
    if token == '(':
        L = []
        while tokens and tokens[0] != ')':
            item = parse_expr(tokens)
            if item is not None:
                L.append(item)
        if tokens and tokens[0] == ')':
            tokens.pop(0)  # remove ')'
        return L
    elif token == ')':
        # stray closing paren: simply ignore it
        return None
    else:
        if token.startswith('"') and token.endswith('"'):
            return token[1:-1]
        try:
            return float(token)
        except ValueError:
            return token

def parse_sexpr(text):
    tokens = tokenize(text)
    expr = []
    while tokens:
        result = parse_expr(tokens)
        if result is not None:
            expr.append(result)
    return expr

# Data container for a symbol definition.
class SymbolData:
    def __init__(self):
        self.lib_id = None
        self.at = None  # tuple (x, y, angle) if provided
        self.reference = None
        self.visuals = []  # list of visual elements

    def __str__(self):
        return f"Symbol(lib_id={self.lib_id}, at={self.at}, reference={self.reference}, visuals={self.visuals})"

def extract_symbols(sexprs):
    symbols = []
    def walk(expr):
        if isinstance(expr, list) and len(expr) > 0:
            # Check if this list represents a symbol
            if isinstance(expr[0], str) and expr[0] == "symbol":
                sym = SymbolData()
                for item in expr[1:]:
                    if isinstance(item, list) and len(item) > 0:
                        key = item[0]
                        if key == "lib_id" and len(item) >= 2:
                            sym.lib_id = item[1]
                        elif key == "at" and len(item) >= 3:
                            x = item[1] if isinstance(item[1], (float, int)) else None
                            y = item[2] if isinstance(item[2], (float, int)) else None
                            angle = item[3] if len(item) >= 4 and isinstance(item[3], (float, int)) else 0
                            sym.at = (x, y, angle)
                        elif key == "property" and len(item) >= 3:
                            if item[1] == "Reference":
                                sym.reference = item[2]
                        elif key in ("polyline", "arc", "text"):
                            sym.visuals.append({"type": key, "data": item})
                    if isinstance(item, list):
                        walk(item)
                symbols.append(sym)
            else:
                for part in expr:
                    walk(part)
    for e in sexprs:
        walk(e)
    return symbols

def render_symbols(symbols):
    scene = QGraphicsScene()
    pen = QPen(QColor("black"))
    brush = QBrush(QColor("lightgray"))
    for sym in symbols:
        if sym.at:
            # Multiply x and y by 10
            x, y, angle = sym.at
            x *= 10
            y *= 10
            ellipse = QGraphicsEllipseItem(x - 5, -y - 5, 10, 10)  # invert y for scene coords
            ellipse.setPen(pen)
            ellipse.setBrush(brush)
            scene.addItem(ellipse)
            if sym.reference:
                text_item = QGraphicsTextItem(sym.reference)
                text_item.setPos(x + 8, -y - 8)
                scene.addItem(text_item)
            for v in sym.visuals:
                if v["type"] == "text":
                    for sub in v["data"]:
                        if isinstance(sub, list) and sub and sub[0] == "at" and len(sub) >= 3:
                            tx = sub[1] * 10
                            ty = sub[2] * 10
                            titem = QGraphicsTextItem("T")
                            titem.setDefaultTextColor(QColor("blue"))
                            titem.setPos(tx + 8, -ty - 8)
                            scene.addItem(titem)
                elif v["type"] == "polyline":
                    pts = []
                    for sub in v["data"]:
                        if isinstance(sub, list) and sub and sub[0] == "pts":
                            for point in sub[1:]:
                                if isinstance(point, list) and point and point[0] == "xy" and len(point) >= 3:
                                    # Multiply x, y by 10 and invert y.
                                    pts.append(QPointF(point[1] * 10, -point[2] * 10))
                    if pts:
                        poly = QPolygonF(pts)
                        poly_item = QGraphicsPolygonItem(poly)
                        poly_item.setPen(QPen(QColor("green")))
                        scene.addItem(poly_item)
                elif v["type"] == "arc":
                    start, mid, end = None, None, None
                    for sub in v["data"]:
                        if isinstance(sub, list) and sub:
                            if sub[0] == "start" and len(sub) >= 3:
                                start = QPointF(sub[1] * 10, -sub[2] * 10)
                            elif sub[0] == "mid" and len(sub) >= 3:
                                mid = QPointF(sub[1] * 10, -sub[2] * 10)
                            elif sub[0] == "end" and len(sub) >= 3:
                                end = QPointF(sub[1] * 10, -sub[2] * 10)
                    if start and mid and end:
                        path = QPainterPath()
                        path.moveTo(start)
                        path.quadTo(mid, end)
                        scene.addPath(path, QPen(QColor("red")))
    return scene

if __name__ == '__main__':
    with open("Skywalker.kicad_sch", "r", encoding="utf-8") as f:
        filetext = f.read()
    sexprs = parse_sexpr(filetext)
    symbols = extract_symbols(sexprs)

    print("Symbols found:")
    for sym in symbols:
        print(f"  Lib: {sym.lib_id}, Ref: {sym.reference}, At: {sym.at}")

    app = QApplication(sys.argv)
    scene = render_symbols(symbols)
    view = QGraphicsView(scene)
    view.setRenderHint(view.renderHints())
    view.setWindowTitle("Skywalker.kicad_sch Render")
    view.resize(800, 600)
    view.show()
    sys.exit(app.exec())