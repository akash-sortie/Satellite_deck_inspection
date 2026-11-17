import sys
import os
import subprocess
import pandas as pd

# --- MATPLOTLIB MUST IMPORT FIRST TO AVOID CRASH ---
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# --- PYQT6 IMPORTS ---
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QLabel, QTableWidget, 
                               QTableWidgetItem, QHeaderView, QSplitter, QFrame)
from PyQt6.QtGui import QPixmap, QColor
from PyQt6.QtCore import Qt

# --- PREMIUM GLOBAL STYLESHEET (TOKYO NIGHT THEME) ---
TOKYO_NIGHT_QSS = """
* {
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
}
QMainWindow {
    background-color: #1a1b26; 
}
QFrame#DashboardPanel {
    background-color: #24283b; 
    border-radius: 12px;
    border: 1px solid #414868;
}
QFrame#KPICard {
    background-color: #1f2335;
    border-radius: 8px;
    border: 1px solid #292e42;
    padding: 5px;
}
QLabel {
    color: #a9b1d6;
}
QPushButton {
    background-color: #3d59a1;
    color: white;
    border-radius: 6px;
    padding: 14px;
    font-size: 13px;
    font-weight: bold;
    border: 1px solid #414868;
}
QPushButton:hover {
    background-color: #7aa2f7;
    border: 1px solid #8caaee;
    color: #1a1b26;
}
QPushButton:pressed {
    background-color: #2ac3de;
}
QPushButton#BtnCV { background-color: #24283b; border-left: 4px solid #bb9af7; color: #a9b1d6; text-align: left; padding-left: 20px;}
QPushButton#BtnCV:hover { background-color: #292e42; color: white; border-left: 4px solid #c0a8f9;}
QPushButton#BtnML { background-color: #24283b; border-left: 4px solid #7dcfff; color: #a9b1d6; text-align: left; padding-left: 20px;}
QPushButton#BtnML:hover { background-color: #292e42; color: white; border-left: 4px solid #92d8ff;}
QPushButton#BtnLoad { background-color: #24283b; border-left: 4px solid #00e676; color: #a9b1d6; text-align: left; padding-left: 20px;}
QPushButton#BtnLoad:hover { background-color: #292e42; color: white; border-left: 4px solid #69f0ae;}

QTableWidget {
    background-color: #1a1b26;
    color: #c0caf5;
    border: none;
    gridline-color: #292e42;
    border-radius: 8px;
}
QHeaderView::section {
    background-color: #24283b;
    color: #7aa2f7;
    font-weight: bold;
    font-size: 12px;
    border: none;
    padding: 10px;
    border-bottom: 2px solid #1a1b26;
}
QTableWidget::item {
    padding: 5px;
    border-bottom: 1px solid #292e42;
}
QTableWidget::item:selected {
    background-color: #33467c;
    color: #ffffff;
}
"""

class AVIDDashboardPro(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AVID Command Center | Aerospace Vision Inspection")
        self.setGeometry(50, 50, 1600, 900)
        
        self.setStyleSheet(TOKYO_NIGHT_QSS)

        # Main Layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # --- HEADER: KPIs ---
        header_frame = QFrame()
        header_frame.setObjectName("DashboardPanel")
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(20, 15, 20, 15)
        
        self.lbl_title = QLabel("AVID PIPELINE")
        self.lbl_title.setStyleSheet("font-size: 26px; font-weight: 900; color: #00E676; letter-spacing: 2px;")
        
        # FIXED: Properly creating and referencing the KPI cards
        self.card_total = self.create_kpi_card("TOTAL INSPECTED", "--")
        self.card_defects = self.create_kpi_card("DEFECT RATE", "--")
        self.card_status = self.create_kpi_card("SYSTEM STATUS", "AWAITING EXECUTION")
        
        header_layout.addWidget(self.lbl_title)
        header_layout.addStretch()
        header_layout.addWidget(self.card_total)
        header_layout.addSpacing(15)
        header_layout.addWidget(self.card_defects)
        header_layout.addSpacing(15)
        header_layout.addWidget(self.card_status)
        main_layout.addWidget(header_frame)

        # --- BODY: 3-PANE SPLITTER ---
        body_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # PANE 1: Controls (Left)
        control_panel = QFrame()
        control_panel.setObjectName("DashboardPanel")
        control_panel.setFixedWidth(280)
        control_layout = QVBoxLayout(control_panel)
        control_layout.setContentsMargins(20, 25, 20, 25)
        
        title_lbl = QLabel("COMMAND OVERRIDE")
        title_lbl.setStyleSheet("color: #7aa2f7; font-weight: bold; font-size: 12px; letter-spacing: 1px;")
        control_layout.addWidget(title_lbl)
        control_layout.addSpacing(15)
        
        self.btn_cv = QPushButton(" 1. INITIALIZE MACRO-VISION")
        self.btn_cv.setObjectName("BtnCV")
        self.btn_cv.clicked.connect(self.run_cv)

        self.btn_ml = QPushButton(" 2. EXECUTE ML CLASSIFIER")
        self.btn_ml.setObjectName("BtnML")
        self.btn_ml.clicked.connect(self.run_ml)

        self.btn_load = QPushButton(" 3. LOAD ANALYTICS")
        self.btn_load.setObjectName("BtnLoad")
        self.btn_load.clicked.connect(self.load_data)

        control_layout.addWidget(self.btn_cv)
        control_layout.addSpacing(5)
        control_layout.addWidget(self.btn_ml)
        control_layout.addSpacing(5)
        control_layout.addWidget(self.btn_load)
        control_layout.addStretch()

        # PANE 2: Image & Table (Center)
        center_splitter = QSplitter(Qt.Orientation.Vertical)
        
        img_panel = QFrame()
        img_panel.setObjectName("DashboardPanel")
        img_layout = QVBoxLayout(img_panel)
        
        self.img_label = QLabel("RUN PIPELINE TO GENERATE DECK MAP")
        self.img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_label.setStyleSheet("color: #565f89; font-weight: bold; letter-spacing: 1px;")
        img_layout.addWidget(self.img_label)
        
        table_panel = QFrame()
        table_panel.setObjectName("DashboardPanel")
        table_layout = QVBoxLayout(table_panel)
        table_layout.setContentsMargins(0,0,0,0) 
        
        self.table = QTableWidget()
        table_layout.addWidget(self.table)
        
        center_splitter.addWidget(img_panel)
        center_splitter.addWidget(table_panel)
        center_splitter.setSizes([600, 400])

        # PANE 3: Graphical Analysis (Right)
        graph_panel = QFrame()
        graph_panel.setObjectName("DashboardPanel")
        graph_layout = QVBoxLayout(graph_panel)
        graph_layout.setContentsMargins(15, 20, 15, 15)
        
        graph_title = QLabel("ML CLUSTER ANALYSIS (Area vs Circularity)")
        graph_title.setStyleSheet("color: #7aa2f7; font-weight: bold; font-size: 12px; letter-spacing: 1px;")
        graph_layout.addWidget(graph_title)
        
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.fig.patch.set_facecolor('#24283b') 
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#1a1b26') 
        
        self.ax.tick_params(colors='#7aa2f7', labelsize=9)
        for spine in self.ax.spines.values():
            spine.set_color('#414868')
        self.ax.grid(True, color='#292e42', linestyle='--', alpha=0.7)
            
        graph_layout.addWidget(self.canvas)

        body_splitter.addWidget(control_panel)
        body_splitter.addWidget(center_splitter)
        body_splitter.addWidget(graph_panel)
        body_splitter.setSizes([280, 800, 500])

        main_layout.addWidget(body_splitter)

    # FIXED: The architecture of the KPI card ensures titles never disappear
    def create_kpi_card(self, title, value):
        frame = QFrame()
        frame.setObjectName("KPICard")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(15, 10, 15, 10)
        
        lbl_title = QLabel(title)
        lbl_title.setStyleSheet("color: #7aa2f7; font-size: 11px; font-weight: bold; letter-spacing: 1px;")
        
        lbl_val = QLabel(value)
        lbl_val.setStyleSheet("color: white; font-size: 22px; font-weight: 900;")
        
        layout.addWidget(lbl_title)
        layout.addWidget(lbl_val)
        
        # Save a direct reference to the value label so we can update it without touching the title
        frame.value_label = lbl_val 
        return frame

    def run_cv(self):
        self.card_status.value_label.setText("MACRO-VISION ACTIVE")
        self.card_status.value_label.setStyleSheet("color: #bb9af7; font-size: 22px; font-weight: 900;")
        QApplication.processEvents()
        subprocess.run(["python", "AVID/pipeline.py", "run", "--cad", "real_deck.csv", "--image", "data_image.jpeg", "--output", "reports/real_run", "--strategy", "blob"])
        self.card_status.value_label.setText("PIPELINE COMPLETE")

    def run_ml(self):
        self.card_status.value_label.setText("ML PROCESSING...")
        self.card_status.value_label.setStyleSheet("color: #7dcfff; font-size: 22px; font-weight: 900;")
        QApplication.processEvents()
        subprocess.run(["python", "AVID/ml_classifier.py"])
        self.card_status.value_label.setText("CLASSIFICATION DONE")
        self.load_data()

    def load_data(self):
        # Load Image
        img_path = "reports/real_run/inspection_overlay.jpg"
        if os.path.exists(img_path):
            pixmap = QPixmap(img_path)
            self.img_label.setPixmap(pixmap.scaled(self.img_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

        # Load Data & Plot
        csv_path = "reports/ml_classification_report.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            
            # 1. Update KPIs (Targeting the value_label only)
            total = len(df)
            nominal = len(df[df['ml_anomaly_score'] == 1])
            defect_rate = ((total - nominal) / total) * 100
            
            self.card_total.value_label.setText(str(total))
            self.card_defects.value_label.setText(f"{defect_rate:.1f}%")
            self.card_defects.value_label.setStyleSheet("color: #f7768e; font-size: 22px; font-weight: 900;" if defect_rate > 0 else "color: #00e676; font-size: 22px; font-weight: 900;")
            
            self.card_status.value_label.setText("DATA LIVE")
            self.card_status.value_label.setStyleSheet("color: #00e676; font-size: 22px; font-weight: 900;")

            # 2. Populate Table
            self.table.setColumnCount(len(df.columns))
            self.table.setRowCount(len(df.index))
            self.table.setHorizontalHeaderLabels(df.columns)
            
            for i in range(len(df.index)):
                for j in range(len(df.columns)):
                    val = str(df.iat[i, j])
                    item = QTableWidgetItem(val)
                    
                    if df.columns[j] == 'ml_category':
                        if val == 'NOMINAL_AEROSPACE_GRADE':
                            item.setBackground(QColor("#1a3b2b")) 
                            item.setForeground(QColor("#00e676"))
                        elif val == 'BRIGHT_OBSTRUCTION':
                            item.setBackground(QColor("#1b3652")) 
                            item.setForeground(QColor("#2ac3de"))
                        elif val == 'BURR_OR_TEAROUT':
                            item.setBackground(QColor("#4a1f28")) 
                            item.setForeground(QColor("#f7768e"))
                        else:
                            item.setBackground(QColor("#4d3119")) 
                            item.setForeground(QColor("#ff9e64"))
                            
                    self.table.setItem(i, j, item)
            self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

            # 3. Update Graph
            self.ax.clear()
            self.ax.set_facecolor('#1a1b26')
            self.ax.set_xlabel("Defect Area (Pixels)", color='#7aa2f7', fontsize=10, weight='bold')
            self.ax.set_ylabel("Circularity (1.0 = Perfect)", color='#7aa2f7', fontsize=10, weight='bold')
            self.ax.grid(True, color='#292e42', linestyle='--', alpha=0.7)
            
            colors = {
                'NOMINAL_AEROSPACE_GRADE': '#00E676', # Green
                'BRIGHT_OBSTRUCTION': '#2AC3DE',      # Cyan
                'BURR_OR_TEAROUT': '#F7768E',         # Red
                'SURFACE_ANOMALY_ML_ONLY': '#FF9E64'  # Orange
            }
            
            for category, color_code in colors.items():
                subset = df[df['ml_category'] == category]
                if not subset.empty:
                    self.ax.scatter(subset['area'], subset['circularity'], 
                                    c=color_code, label=category, s=90, alpha=0.8, edgecolors='#1a1b26', linewidths=1.5)
            
            legend = self.ax.legend(facecolor='#1f2335', edgecolor='#414868', fontsize=9, loc='upper right')
            for text in legend.get_texts(): text.set_color('#c0caf5')
            
            self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AVIDDashboardPro()
    window.showMaximized()
    sys.exit(app.exec())