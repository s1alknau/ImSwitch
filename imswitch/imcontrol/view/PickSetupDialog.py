import os

from qtpy import QtCore, QtWidgets


class PickSetupDialog(QtWidgets.QDialog):
    """ Dialog for picking the setup to use for imcontrol. """

    def __init__(self, parent=None, *args, **kwargs):
        super().__init__(parent, QtCore.Qt.WindowSystemMenuHint | QtCore.Qt.WindowTitleHint,
                         *args, **kwargs)
        self.setWindowTitle('Select hardware setup')

        # Set when the user picked a different configuration folder, so the
        # caller knows that the options it loaded beforehand are stale.
        self.configFolderChanged = False

        self.informationLabel = QtWidgets.QLabel(
            'Select the configuration file for your hardware setup:'
        )
        self.setupPicker = QtWidgets.QComboBox()

        # Row showing which configuration folder the setups are read from,
        # with a button to switch to a different one.
        self.folderLabel = QtWidgets.QLabel()
        self.folderLabel.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.folderButton = QtWidgets.QPushButton('Change folder…')
        self.folderButton.clicked.connect(self._pickConfigFolder)

        folderLayout = QtWidgets.QHBoxLayout()
        folderLayout.addWidget(self.folderLabel, stretch=1)
        folderLayout.addWidget(self.folderButton)

        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            QtCore.Qt.Horizontal,
            self
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.informationLabel)
        layout.addLayout(folderLayout)
        layout.addWidget(self.setupPicker)
        layout.addWidget(self.buttons)
        self.setLayout(layout)

        self._updateFolderLabel()

    def setSetups(self, setupList):
        self.setupPicker.clear()
        for setup in setupList:
            self.setupPicker.addItem(setup)

        # An empty folder would let the user confirm a selection that does not
        # exist, which fails much later with a confusing message.
        self.buttons.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(bool(setupList))

    def getSelectedSetup(self):
        return self.setupPicker.currentText()

    def setSelectedSetup(self, setup):
        index = self.setupPicker.findText(setup)
        if index > -1:
            self.setupPicker.setCurrentIndex(index)

    def _currentConfigFolder(self):
        from imswitch.imcommon.model import dirtools
        return dirtools.UserFileDirs.Root

    def _updateFolderLabel(self):
        folder = self._currentConfigFolder()
        self.folderLabel.setText(f'Folder: {folder}')
        self.folderLabel.setToolTip(folder)

    def _pickConfigFolder(self):
        """ Switch to a different ImSwitchConfig folder and reload the list. """
        from imswitch.imcommon.model.storage_paths import set_config_path
        from imswitch.imcontrol.model import configfiletools

        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, 'Select ImSwitch configuration folder', self._currentConfigFolder()
        )
        if not folder:
            return

        succeeded, message = set_config_path(folder)
        if not succeeded:
            QtWidgets.QMessageBox.warning(self, 'Configuration folder', message)
            return

        configfiletools.refreshPaths()
        self.configFolderChanged = True
        self._updateFolderLabel()

        setupList = configfiletools.getSetupList()
        self.setSetups(setupList)
        if not setupList:
            QtWidgets.QMessageBox.warning(
                self, 'Configuration folder',
                'No setup files found in\n'
                + os.path.join(folder, 'imcontrol_setups')
                + '\n\nPick a folder that contains an imcontrol_setups directory.'
            )


# Copyright (C) 2020-2024 ImSwitch developers
# This file is part of ImSwitch.
#
# ImSwitch is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ImSwitch is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
