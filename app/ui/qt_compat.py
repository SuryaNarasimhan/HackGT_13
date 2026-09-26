"""
Qt Compatibility Shim for SocialLens
Detects PyQt6, PySide6, or PyQt5, ensuring graceful degradation if not yet installed.
"""

import logging

logger = logging.getLogger(__name__)

QT_AVAILABLE = False
QT_LIB = None
QtCore = None
QtWidgets = None
QtGui = None

try:
    from PyQt6 import QtCore, QtWidgets, QtGui
    QT_AVAILABLE = True
    QT_LIB = "PyQt6"
    logger.info("Using PyQt6 GUI backend.")
except ImportError:
    try:
        from PySide6 import QtCore, QtWidgets, QtGui
        QT_AVAILABLE = True
        QT_LIB = "PySide6"
        logger.info("Using PySide6 GUI backend.")
    except ImportError:
        try:
            from PyQt5 import QtCore, QtWidgets, QtGui
            QT_AVAILABLE = True
            QT_LIB = "PyQt5"
            logger.info("Using PyQt5 GUI backend.")
        except ImportError:
            QT_AVAILABLE = False
            QT_LIB = None
            logger.info("No Qt binding (PyQt6/PySide6/PyQt5) found.")
