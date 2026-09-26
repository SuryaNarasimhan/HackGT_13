"""
Pytest configuration and session fixtures for SocialLens test suite.
Initializes headless QApplication for Qt widget tests.
"""

import pytest
from app.ui.qt_compat import QT_AVAILABLE, QtWidgets


@pytest.fixture(scope="session", autouse=True)
def qapp():
    """Ensures a QApplication instance exists for all UI tests in headless offscreen mode."""
    if QT_AVAILABLE and QtWidgets is not None:
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(["--platform", "offscreen"])
        yield app
    else:
        yield None
