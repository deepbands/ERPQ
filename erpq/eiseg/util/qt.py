import os.path as osp
import qtpy.QtGui as QtGui

here = osp.dirname(osp.abspath(__file__))


def newIcon(icon):
    if isinstance(icon, list) or isinstance(icon, tuple):
        pixmap = QtGui.QPixmap(100, 100)
        c = icon
        pixmap.fill(QtGui.QColor(c[0], c[1], c[2]))
        return QtGui.QIcon(pixmap)
    icons_dir = here.replace("util", "resource")
    path = osp.join(icons_dir, f"{icon}.png")
    return QtGui.QIcon(path)