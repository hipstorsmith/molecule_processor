import sys
import os
from markdown import markdown
from glob import glob
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QDesktopServices
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from interpolator import run_interpolation
from vec_changer import vec_changer
from method_changer import method_changer
from molecule_delimiter import molecules_delimiter
from energy_profile_plotter import energy_profile


def resource_path(*parts: str) -> str:
    """
    Build an absolute path to bundled resources that works both:
    - in source checkout (runs from repo)
    - in PyInstaller --onefile (resources extracted to sys._MEIPASS)
    """
    base = getattr(sys, "_MEIPASS", os.path.abspath(os.path.dirname(__file__)))
    return os.path.join(base, *parts)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi(resource_path('design', 'mainWindow.ui'), self)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)

        # Wire buttons to their helpers
        #  Interpolator
        self.interpolationInitialFileButton.clicked.connect(
            lambda: self._get_file(self.interpolationInitialFileEdit, "XYZ files (*.xyz)"))
        self.interpolationTransformedFileButton.clicked.connect(
            lambda: self._get_file(self.interpolationTransformedFileEdit, "XYZ files (*.xyz)"))
        self.interpolationOutputFolderButton.clicked.connect(
            lambda: self._get_dir(self.interpolationOutputFolderEdit))
        self.interpolatorOpenFolderButton.clicked.connect(self._open_interpolation_output_folder)
        self.interpolationRunButton.clicked.connect(self._run_interpolation)

        # Change $VEC group
        self.vecInputFileButton.clicked.connect(lambda: self._get_inp_or_dir(self.vecInputFileEdit))
        self.vecVecFileButton.clicked.connect(lambda: self._get_file(self.vecVecFileEdit, "All Files (*)"))
        self.vecOutputFolderButton.clicked.connect(lambda: self._get_dir(self.vecOutputFolderEdit))
        self.vecChangeVecButton.clicked.connect(self._run_vec_changer)

        # Change method group
        self.methodOutputFolderButton.clicked.connect(lambda: self._get_dir(self.methodOutputFolderEdit))
        self.methodInputFileButton.clicked.connect(lambda: self._get_inp_or_dir(self.methodInputFileEdit))
        self.methodMethodFileButton.clicked.connect(lambda: self._get_file(self.methodMethodFileEdit, "All Files (*)"))
        self.vecChangeVecButton.clicked.connect(self._run_method_changer)

        # Delimiter group
        self.delimiterXyzFileButton.clicked.connect(
            lambda: self._get_file(self.delimiterXyzFileEdit, "XYZ files (*.xyz)"))
        self.delimiterOutputFolderButton.clicked.connect(lambda: self._get_dir(self.delimiterOutputFolderEdit))
        self.delimiterMergeToDimers.stateChanged.connect(self._enable_dimer_options)
        self.delimiterSplitButton.clicked.connect(self._run_molecules_delimiter)

        self.plotRunEditorButton.clicked.connect(self._open_plot_window)
        self.actionExit.triggered.connect(QtWidgets.qApp.quit)

        # 2) Help menu
        self.actionAbout.triggered.connect(self._show_help)

    def _get_file(self, line_edit: QtWidgets.QLineEdit, filter_str: str):
        """Open a file dialog, restrict by filter, set path if chosen."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select File",
            "",
            filter_str
        )
        if path:
            line_edit.setText(path)

    def _get_dir(self, line_edit: QtWidgets.QLineEdit):
        """Open a folder dialog and set the chosen path."""
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select Folder",
            ""
        )
        if path:
            line_edit.setText(path)

    def _get_inp_or_dir(self, line_edit: QtWidgets.QLineEdit):
        """
        First try picking an .inp file; if the user cancels,
        fall back to picking a folder.
        """
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select .inp File (or cancel for folder)",
            "",
            "INP files (*.inp)"
        )
        if not path:
            path = QtWidgets.QFileDialog.getExistingDirectory(
                self,
                "Select Folder",
                ""
            )
        if path:
            line_edit.setText(path)

    def _open_folder(self, path: str):
        if not path or not os.path.isdir(path):
            QtWidgets.QMessageBox.warning(self, "Invalid Folder",
                                          f"“{path}” is not a valid directory.")
            return
        url = QUrl.fromLocalFile(path)
        QDesktopServices.openUrl(url)

    def _open_interpolation_output_folder(self):
        self._open_folder(self.interpolationOutputFolderEdit.text())

    def _safe_call(self, func, *args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))
        else:
            QtWidgets.QMessageBox.information(self, "Done",
                                              f"{func.__name__} completed successfully.")

    def _enable_dimer_options(self):
        if self.delimiterMergeToDimers.isChecked():
            self.delimiterDropDuplicates.setEnabled(True)
            self.delimiterContactDistance.setEnabled(True)
        else:
            self.delimiterDropDuplicates.setDisabled(True)
            self.delimiterContactDistance.setDisabled(True)

    def _run_interpolation(self):
        kwargs = {
            'xyz_file_init': self.interpolationInitialFileEdit.text(),
            'xyz_file_trans': self.interpolationTransformedFileEdit.text(),
            'zmt_folder_out': self.interpolationOutputFolderEdit.text(),
            'n_points': self.interpolationStepsAmountBox.value()
        }
        self._safe_call(run_interpolation, **kwargs)

    def _run_vec_changer(self):
        input_path = self.vecInputFileEdit.text()
        vec_file = self.vecVecFileEdit.text()
        output_path = self.vecOutputFolderEdit.text()
        if os.path.isdir(input_path):
            for input_file in glob(os.path.join(input_path, '*.inp')):
                output_file = os.path.join(output_path, os.path.basename(input_file))
                self._safe_call(vec_changer, input_file=input_file, vec_file=vec_file, output_file=output_file)
        else:
            output_file = os.path.join(output_path, os.path.basename(input_path))
            self._safe_call(vec_changer, input_file=input_path, vec_file=vec_file, output_file=output_file)

    def _run_method_changer(self):
        input_path = self.methodInputFileEdit.text()
        method_file = self.methodMethodFileEdit.text()
        output_path = self.methodOutputFolderEdit.text()
        if os.path.isdir(input_path):
            for input_file in glob(os.path.join(input_path, '*.inp')):
                output_file = os.path.join(output_path, os.path.basename(input_file))
                self._safe_call(method_changer, input_file=input_file, method_file=method_file, output_file=output_file)
        else:
            output_file = os.path.join(output_path, os.path.basename(input_path))
            self._safe_call(method_changer, input_file=input_path, method_file=method_file, output_file=output_file)

    def _run_molecules_delimiter(self):
        kwargs = {
            'input_file': self.delimiterXyzFileEdit.text(),
            'output_folder': self.delimiterOutputFolderEdit.text(),
            'min_atoms_per_molecule': self.delimiterMinAtomsBox.value(),
            'max_bond_length': self.delimiterBondAtomsBox.value(),
            'merge_to_dimers': self.delimiterMergeToDimers.isChecked(),
            'contact_distance': self.delimiterContactDistance.value(),
            'drop_duplicates': self.delimiterDropDuplicates.isChecked()
        }
        self._safe_call(molecules_delimiter, **kwargs)

    def _open_plot_window(self):
        self.plot_window = PlotWindow(self)  # store on self to keep it alive
        self.plot_window.show()

    def _show_help(self):
        # keep a ref so it doesn't get garbage-collected
        self._help_dialog = HelpDialog(self)
        self._help_dialog.exec_()  # Modal; use .show() for non-modal


class PlotWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        uic.loadUi(resource_path('design', 'plotSettingsWindow.ui'), self)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)

        self._fig = Figure()
        self._canvas = FigureCanvas(self._fig)

        margin = self.plotCanvasContainer.lineWidth() + self.plotCanvasContainer.midLineWidth()

        layout = QtWidgets.QVBoxLayout(self.plotCanvasContainer)
        layout.setContentsMargins(margin, margin, margin, margin)
        layout.addWidget(self._canvas)

        self.qdptFIlesButton.clicked.connect(lambda: self._get_dir(self.qdptFilesEdit))
        self.buildPlotButton.clicked.connect(self._build_plot)

    def _get_dir(self, line_edit: QtWidgets.QLineEdit):
        """Open a folder dialog and set the chosen path."""
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select Folder",
            ""
        )
        if path:
            line_edit.setText(path)

    def _build_plot(self):
        # self._fig = Figure()
        # self._canvas = FigureCanvas(self._fig)

        try:
            ereorg, mingap, delta_g = energy_profile(
                folder_path=self.qdptFilesEdit.text(),
                line_1_style=self.line1StyleBox.currentText(),
                line_2_style=self.line2StyleBox.currentText(),
                line_1_width=self.line1WidthBox.value(),
                line_2_width=self.line2WidthBox.value(),
                line_1_color=self.line1ColorBox.currentText(),
                line_2_color=self.line2ColorBox.currentText(),
                line_1_marker=self.line1MarkerStyleBox.currentText(),
                line_2_marker=self.line2MarkerStyleBox.currentText(),
                line_1_marker_size=self.line1MarkerSizeBox.value(),
                line_2_marker_size=self.line2MarkerSizeBox.value(),
                axis_line_width=self.axisWidthBox.value(),
                label_font_size=self.labelFontSizeBox.value(),
                x_min=self.xMinBox.value(),
                y_min=self.yMinBox.value(),
                x_max=self.xMaxBox.value(),
                y_max=self.yMaxBox.value(),
                plot_title=self.plotTitleEdit.text(),
                fig=self._fig,
                canvas=self._canvas
            )
            self.ereorgEdit.setText(str(ereorg))
            self.deltaGEdit.setText(str(delta_g))
            self.energyGapEdit.setText(str(mingap))
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))


class HelpDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        uic.loadUi(resource_path('design', 'helpWindow.ui'), self)

        # 1) Read the markdown file
        md_path = os.path.join(os.path.dirname(__file__), 'README.md')
        try:
            with open(md_path, 'r', encoding='utf-8') as f:
                md_text = f.read()
        except IOError:
            self.textBrowserHelp.setPlainText("Could not load README.")
            return

        # 2) Convert to HTML and display
        html = markdown(md_text, extensions=['fenced_code', 'tables'])
        self.helpTextBrowser.setHtml(html)
