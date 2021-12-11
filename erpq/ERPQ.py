# -*- coding: utf-8 -*-
"""
/***************************************************************************
 ERPQ
                                 A QGIS plugin
 EISeg remote sensing plug-in in QGIS
 Generated by Plugin Builder: http://g-sherman.github.io/Qgis-Plugin-Builder/
                              -------------------
        begin                : 2021-11-03
        git sha              : $Format:%H$
        copyright            : (C) 2021 by geoyee
        email                : geoyee@yeah.net
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""
from qgis.PyQt.QtCore import QSettings, QTranslator, QCoreApplication, Qt
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction, QFileDialog
# TODO : add scikit-image to req
try:
    import qgis.PyQt.QtGui as QtGui
    import qgis.PyQt.QtWidgets as QtWidgets
    from qgis.PyQt.QtWidgets import QMessageBox, QTableWidgetItem
except:
    import qtpy.QtGui as QtGui
    import qtpy.QtWidgets as QtWidgets
    from qtpy.QtWidgets import QMessageBox, QTableWidgetItem
# Initialize Qt resources from file resources.py
from .resources import *

# Import the code for the DockWidget
from .ERPQ_dockwidget import ERPQDockWidget
import os.path as osp

# EISEG
from .eiseg.controller import InteractiveController
from .eiseg.util.colormap import colorMap
from .eiseg.util.qt import newIcon


class ERPQ:
    """QGIS Plugin Implementation."""

    def __init__(self, iface):
        """Constructor.

        :param iface: An interface instance that will be passed to this class
            which provides the hook by which you can manipulate the QGIS
            application at run time.
        :type iface: QgsInterface
        """
        # Save reference to the QGIS interface
        self.iface = iface

        # initialize plugin directory
        self.plugin_dir = osp.dirname(__file__)

        # initialize locale
        locale = QSettings().value('locale/userLocale')[0:2]
        locale_path = osp.join(
            self.plugin_dir,
            'i18n',
            'ERPQ_{}.qm'.format(locale))

        if osp.exists(locale_path):
            self.translator = QTranslator()
            self.translator.load(locale_path)
            QCoreApplication.installTranslator(self.translator)

        # Declare instance attributes
        self.actions = []
        self.menu = self.tr(u'&ERPQ')
        # TODO: We are going to let the user set this up in a future iteration
        self.toolbar = self.iface.addToolBar(u'ERPQ')
        self.toolbar.setObjectName(u'ERPQ')

        #print "** INITIALIZING ERPQ"

        self.pluginIsActive = False
        self.dockwidget = None

        # TODO: DIY
        self.predictor_params = {
            "brs_mode": "NoBRS",
            "with_flip": False,
            "zoom_in_params": {
                "skip_clicks": -1,
                "target_size": (400, 400),
                "expansion_ratio": 1.4,
            },
            "predictor_params": {
                "net_clicks_limit": None,
                "max_size": 800,
                "with_mask": True,
            },
        }
        self.colorMap = colorMap


    # noinspection PyMethodMayBeStatic
    def tr(self, message):
        """Get the translation for a string using Qt translation API.

        We implement this ourselves since we do not inherit QObject.

        :param message: String for translation.
        :type message: str, QString

        :returns: Translated version of message.
        :rtype: QString
        """
        # noinspection PyTypeChecker,PyArgumentList,PyCallByClass
        return QCoreApplication.translate('ERPQ', message)


    def add_action(
        self,
        icon_path,
        text,
        callback,
        enabled_flag=True,
        add_to_menu=True,
        add_to_toolbar=True,
        status_tip=None,
        whats_this=None,
        parent=None):
        """Add a toolbar icon to the toolbar.

        :param icon_path: Path to the icon for this action. Can be a resource
            path (e.g. ':/plugins/foo/bar.png') or a normal file system path.
        :type icon_path: str

        :param text: Text that should be shown in menu items for this action.
        :type text: str

        :param callback: Function to be called when the action is triggered.
        :type callback: function

        :param enabled_flag: A flag indicating if the action should be enabled
            by default. Defaults to True.
        :type enabled_flag: bool

        :param add_to_menu: Flag indicating whether the action should also
            be added to the menu. Defaults to True.
        :type add_to_menu: bool

        :param add_to_toolbar: Flag indicating whether the action should also
            be added to the toolbar. Defaults to True.
        :type add_to_toolbar: bool

        :param status_tip: Optional text to show in a popup when mouse pointer
            hovers over the action.
        :type status_tip: str

        :param parent: Parent widget for the new action. Defaults None.
        :type parent: QWidget

        :param whats_this: Optional text to show in the status bar when the
            mouse pointer hovers over the action.

        :returns: The action that was created. Note that the action is also
            added to self.actions list.
        :rtype: QAction
        """

        icon = QIcon(icon_path)
        action = QAction(icon, text, parent)
        action.triggered.connect(callback)
        action.setEnabled(enabled_flag)

        if status_tip is not None:
            action.setStatusTip(status_tip)

        if whats_this is not None:
            action.setWhatsThis(whats_this)

        if add_to_toolbar:
            self.toolbar.addAction(action)

        if add_to_menu:
            self.iface.addPluginToMenu(
                self.menu,
                action)

        self.actions.append(action)

        return action


    def initGui(self):
        """Create the menu entries and toolbar icons inside the QGIS GUI."""

        icon_path = ':/plugins/ERPQ/icon.png'
        self.add_action(
            icon_path,
            text=self.tr(u'ERPQ tools'),
            callback=self.run,
            parent=self.iface.mainWindow())

    #--------------------------------------------------------------------------

    def onClosePlugin(self):
        """Cleanup necessary items here when plugin dockwidget is closed"""

        #print "** CLOSING ERPQ"

        # disconnects
        self.dockwidget.closingPlugin.disconnect(self.onClosePlugin)

        # remove this statement if dockwidget is to remain
        # for reuse if plugin is reopened
        # Commented next statement since it causes QGIS crashe
        # when closing the docked window:
        # self.dockwidget = None

        self.pluginIsActive = False


    def unload(self):
        """Removes the plugin menu item and icon from QGIS GUI."""

        #print "** UNLOAD ERPQ"

        for action in self.actions:
            self.iface.removePluginMenu(
                self.tr(u'&ERPQ'),
                action)
            self.iface.removeToolBarIcon(action)
        # remove the toolbar
        del self.toolbar

    def loadParams(self):
        filters = self.tr("Paddle静态模型权重文件(*.pdiparams)")
        param_path, _ = QFileDialog.getOpenFileName(
            self.dockwidget, self.tr("选择模型参数"), "", filters)
        success, res = self.controller.setModel(param_path, False)
        if success:
            self.dockwidget.edtParams.setText(param_path.split("/")[-1].split(".")[0])
        else:
            self.dockwidget.edtParams.setText(res)

    # TODO: DIY
    def sldThresholdChanged(self):
        self.dockwidget.labThreshold.setText(str(self.segThreshold))
        if not self.controller or self.controller.image is None:
            return
        self.controller.prob_thresh = self.segThreshold
        # self.updateImage()

    @property
    def segThreshold(self):
        return self.dockwidget.sldThreshold.value() / 100.

    def initLabelTab(self):
        self.dockwidget.tabLabel.horizontalHeader().hide()
        # 铺满
        self.dockwidget.tabLabel.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.Stretch)
        self.dockwidget.tabLabel.verticalHeader().hide()
        self.dockwidget.tabLabel.setColumnWidth(0, 10)
        # self.dockwidget.tabLabel.setMinimumWidth()
        self.dockwidget.tabLabel.clearContents()
        self.dockwidget.tabLabel.setRowCount(0)
        self.dockwidget.tabLabel.setColumnCount(4)

    def addLabel(self):
        c = self.colorMap.get_color()
        table = self.dockwidget.tabLabel
        idx = table.rowCount()
        table.insertRow(table.rowCount())
        self.controller.addLabel(idx + 1, "", c)
        numberItem = QTableWidgetItem(str(idx + 1))
        numberItem.setFlags(QtCore.Qt.ItemIsEnabled)
        table.setItem(idx, 0, numberItem)
        table.setItem(idx, 1, QTableWidgetItem())
        colorItem = QTableWidgetItem()
        colorItem.setBackground(QtGui.QColor(c[0], c[1], c[2]))
        colorItem.setFlags(QtCore.Qt.ItemIsEnabled)
        table.setItem(idx, 2, colorItem)
        delItem = QTableWidgetItem()
        delItem.setIcon(newIcon("Clear"))
        delItem.setTextAlignment(Qt.AlignCenter)
        delItem.setFlags(QtCore.Qt.ItemIsEnabled)
        table.setItem(idx, 3, delItem)
        self.adjustTableSize()
        self.labelListClicked(self.dockwidget.tabLabel.rowCount() - 1, 0)

    def adjustTableSize(self):
        self.dockwidget.tabLabel.horizontalHeader().setDefaultSectionSize(25)
        self.dockwidget.tabLabel.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.Fixed)
        self.dockwidget.tabLabel.horizontalHeader().setSectionResizeMode(
            3, QtWidgets.QHeaderView.Fixed)
        self.dockwidget.tabLabel.horizontalHeader().setSectionResizeMode(
            2, QtWidgets.QHeaderView.Fixed)
        self.dockwidget.tabLabel.setColumnWidth(2, 50)

    def clearLabelList(self):
        if len(self.controller.labelList) == 0:
            return True
        res = self.warn(
            self.tr("清空标签列表?"),
            self.tr("请确认是否要清空标签列表"),
            QMessageBox.Yes | QMessageBox.Cancel,
        )
        if res == QMessageBox.Cancel:
            return False
        self.controller.labelList.clear()
        if self.controller:
            self.controller.label_list = []
            self.controller.curr_label_number = 0
        self.dockwidget.tabLabel.clear()
        self.dockwidget.tabLabel.setRowCount(0)
        return True

    def labelListDoubleClick(self, row, col):
        if col != 2:
            return
        table = self.dockwidget.tabLabel
        color = QtWidgets.QColorDialog.getColor()
        if color.getRgb() == (0, 0, 0, 255):
            return
        table.item(row, col).setBackground(color)
        self.controller.labelList[row].color = color.getRgb()[:3]
        if self.controller:
            self.controller.label_list = self.controller.labelList
        # TODO: 颜色
        # for p in self.scene.polygon_items:
        #     idlab = self.controller.labelList.getLabelById(p.labelIndex)
        #     if idlab is not None:
        #         color = idlab.color
        #         p.setColor(color, color)
        self.labelListClicked(row, 0)

    @property
    def currLabelIdx(self):
        return self.controller.curr_label_number - 1

    def labelListClicked(self, row, col):
        table = self.dockwidget.tabLabel
        if col == 3:
            labelIdx = int(table.item(row, 0).text())
            self.controller.labelList.remove(labelIdx)
            table.removeRow(row)
        if col == 0 or col == 1:
            for cl in range(2):
                for idx in range(len(self.controller.labelList)):
                    table.item(idx, cl).setBackground(QtGui.QColor(255, 255, 255))
                table.item(row, cl).setBackground(QtGui.QColor(48, 140, 198))
                table.item(row, 0).setSelected(True)
            if self.controller:
                self.controller.setCurrLabelIdx(int(table.item(row, 0).text()))
                self.controller.label_list = self.controller.labelList

    def labelListItemChanged(self, row, col):
        self.colorMap.usedColors = self.controller.labelList.colors
        try:
            if col == 1:
                name = self.dockwidget.tabLabel.item(row, col).text()
                self.controller.labelList[row].name = name
        except:
            pass

    #--------------------------------------------------------------------------

    def run(self):
        """Run method that loads and starts the plugin"""

        if not self.pluginIsActive:
            self.pluginIsActive = True

            #print "** STARTING ERPQ"

            # dockwidget may not exist if:
            #    first run of plugin
            #    removed on close (see self.onClosePlugin method)
            if self.dockwidget == None:
                # Create the dockwidget (after translation) and keep reference
                self.dockwidget = ERPQDockWidget()
                self.initLabelTab()  # 初始化tab
                # 控制器
                self.controller = InteractiveController(
                    predictor_params=self.predictor_params,
                    prob_thresh=self.segThreshold)

            # connect to provide cleanup on closing of dockwidget
            self.dockwidget.closingPlugin.connect(self.onClosePlugin)
            # 按钮
            self.dockwidget.btnParams.clicked.connect(self.loadParams)
            self.dockwidget.btnAddLabel.clicked.connect(self.addLabel)
            # 滑块
            self.dockwidget.sldThreshold.valueChanged.connect(self.sldThresholdChanged)
            # 标签
            self.dockwidget.tabLabel.cellDoubleClicked.connect(self.labelListDoubleClick)
            self.dockwidget.tabLabel.cellClicked.connect(self.labelListClicked)
            self.dockwidget.tabLabel.cellChanged.connect(self.labelListItemChanged)

            # show the dockwidget
            # TODO: fix to allow choice of dock location
            self.iface.addDockWidget(Qt.LeftDockWidgetArea, self.dockwidget)
            self.dockwidget.show()
