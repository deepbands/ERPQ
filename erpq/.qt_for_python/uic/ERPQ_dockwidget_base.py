# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'c:\Users\Youss\Documents\pp\New folder\ERPQ\erpq\ERPQ_dockwidget_base.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_ERPQDockWidgetBase(object):
    def setupUi(self, ERPQDockWidgetBase):
        ERPQDockWidgetBase.setObjectName("ERPQDockWidgetBase")
        ERPQDockWidgetBase.resize(450, 380)
        self.dockWidgetContents = QtWidgets.QWidget()
        self.dockWidgetContents.setObjectName("dockWidgetContents")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.dockWidgetContents)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.vLayDock = QtWidgets.QVBoxLayout()
        self.vLayDock.setObjectName("vLayDock")
        self.hLayParams = QtWidgets.QHBoxLayout()
        self.hLayParams.setObjectName("hLayParams")
        self.labParams = QtWidgets.QLabel(self.dockWidgetContents)
        self.labParams.setObjectName("labParams")
        self.hLayParams.addWidget(self.labParams)
        self.edtParams = QtWidgets.QLineEdit(self.dockWidgetContents)
        self.edtParams.setReadOnly(True)
        self.edtParams.setObjectName("edtParams")
        self.hLayParams.addWidget(self.edtParams)
        self.btnParams = QtWidgets.QPushButton(self.dockWidgetContents)
        self.btnParams.setMinimumSize(QtCore.QSize(40, 0))
        self.btnParams.setMaximumSize(QtCore.QSize(40, 16777215))
        self.btnParams.setObjectName("btnParams")
        self.hLayParams.addWidget(self.btnParams)
        self.vLayDock.addLayout(self.hLayParams)
        self.vLayLabel = QtWidgets.QVBoxLayout()
        self.vLayLabel.setObjectName("vLayLabel")
        self.hLayLabel = QtWidgets.QHBoxLayout()
        self.hLayLabel.setObjectName("hLayLabel")
        self.labLabel = QtWidgets.QLabel(self.dockWidgetContents)
        self.labLabel.setObjectName("labLabel")
        self.hLayLabel.addWidget(self.labLabel)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.hLayLabel.addItem(spacerItem)
        self.btnAddLabel = QtWidgets.QPushButton(self.dockWidgetContents)
        self.btnAddLabel.setMinimumSize(QtCore.QSize(40, 0))
        self.btnAddLabel.setMaximumSize(QtCore.QSize(40, 16777215))
        self.btnAddLabel.setObjectName("btnAddLabel")
        self.hLayLabel.addWidget(self.btnAddLabel)
        self.vLayLabel.addLayout(self.hLayLabel)
        self.tabLabel = QtWidgets.QTableWidget(self.dockWidgetContents)
        self.tabLabel.setObjectName("tabLabel")
        self.tabLabel.setColumnCount(0)
        self.tabLabel.setRowCount(0)
        self.vLayLabel.addWidget(self.tabLabel)
        self.vLayDock.addLayout(self.vLayLabel)
        self.vLayThreshold = QtWidgets.QVBoxLayout()
        self.vLayThreshold.setObjectName("vLayThreshold")
        self.hLayThreshold = QtWidgets.QHBoxLayout()
        self.hLayThreshold.setObjectName("hLayThreshold")
        self.labThresholdName = QtWidgets.QLabel(self.dockWidgetContents)
        self.labThresholdName.setObjectName("labThresholdName")
        self.hLayThreshold.addWidget(self.labThresholdName)
        self.labThreshold = QtWidgets.QLabel(self.dockWidgetContents)
        self.labThreshold.setObjectName("labThreshold")
        self.hLayThreshold.addWidget(self.labThreshold)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.hLayThreshold.addItem(spacerItem1)
        self.vLayThreshold.addLayout(self.hLayThreshold)
        self.sldThreshold = QtWidgets.QSlider(self.dockWidgetContents)
        self.sldThreshold.setMaximum(100)
        self.sldThreshold.setSingleStep(1)
        self.sldThreshold.setProperty("value", 50)
        self.sldThreshold.setOrientation(QtCore.Qt.Horizontal)
        self.sldThreshold.setObjectName("sldThreshold")
        self.vLayThreshold.addWidget(self.sldThreshold)
        self.vLayDock.addLayout(self.vLayThreshold)
        self.horizontalLayout_4.addLayout(self.vLayDock)
        ERPQDockWidgetBase.setWidget(self.dockWidgetContents)

        self.retranslateUi(ERPQDockWidgetBase)
        QtCore.QMetaObject.connectSlotsByName(ERPQDockWidgetBase)

    def retranslateUi(self, ERPQDockWidgetBase):
        _translate = QtCore.QCoreApplication.translate
        ERPQDockWidgetBase.setWindowTitle(_translate("ERPQDockWidgetBase", "EISeg in QGIS"))
        self.labParams.setText(_translate("ERPQDockWidgetBase", "Model Parameter Selection"))
        self.btnParams.setText(_translate("ERPQDockWidgetBase", "..."))
        self.labLabel.setText(_translate("ERPQDockWidgetBase", "Label List"))
        self.btnAddLabel.setText(_translate("ERPQDockWidgetBase", "+"))
        self.labThresholdName.setText(_translate("ERPQDockWidgetBase", "Segmentation Threshold:"))
        self.labThreshold.setText(_translate("ERPQDockWidgetBase", "0.5"))
