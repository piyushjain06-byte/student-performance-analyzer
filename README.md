import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter import font as tkfont
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.linear_model import LinearRegression
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.units import inch
import io
import json
import os
from datetime import datetime


class StudentPerformanceAnalyzer:
    def _init_(self):
        self.root = tk.Tk()
        self.root.title("Student Performance Analyzer")
        self.center_window(720, 520)
        self.root.configure(bg='#1f2937')

        # Data storage
        self.students_data = None
        self.current_user = None
        self.user_type = None
        self.teachers_comments = {}

        # Theming and styles
        style = ttk.Style()
        try:
            style.theme_use('clam')
        except Exception:
            pass

        self.primary_color = '#2563eb'     # blue-600
        self.danger_color = '#ef4444'      # red-500
        self.success_color = '#10b981'     # emerald-500
        self.warning_color = '#f59e0b'     # amber-500
        self.accent_color = '#8b5cf6'      # violet-500
        self.surface_color = '#111827'     # gray-900
        self.card_color = '#0b1220'        # very dark card
        self.text_on_dark = '#e5e7eb'      # gray-200

        self.ui_font = tkfont.nametofont('TkDefaultFont')
        self.ui_font.configure(family='Segoe UI', size=10)
        self.h1_font = ('Segoe UI', 28, 'bold')
        self.h2_font = ('Segoe UI', 14, 'bold')

        style.configure('TButton', font=('Segoe UI', 11, 'bold'), padding=10)
        style.map('TButton',
                  background=[('active', '#1d4ed8')],
                  foreground=[('disabled', '#9ca3af')])
        style.configure('Primary.TButton', background=self.primary_color, foreground='white')
        style.configure('Danger.TButton', background=self.danger_color, foreground='white')
        style.configure('Success.TButton', background=self.success_color, foreground='white')
        style.configure('Warn.TButton', background=self.warning_color, foreground='black')

        style.configure('Card.TFrame', background=self.card_color)
        style.configure('Surface.TFrame', background=self.surface_color)
        style.configure('TLabel', background=self.surface_color, foreground=self.text_on_dark, font=('Segoe UI', 10))

        # Load existing comments if available
        self.load_comments()

        self.show_login_page()

    def center_window(self, width, height):
        self.root.update_idletasks()
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        x = int((screen_w - width) / 2)
        y = int((screen_h - height) / 2.5)
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        self.root.minsize(640, 480)

    def load_comments(self):
        """Load teacher comments from file"""
        if os.path.exists('teacher_comments.json'):
            with open('teacher_comments.json', 'r') as f:
                self.teachers_comments = json.load(f)

    def save_comments(self):
        """Save teacher comments to file"""
        with open('teacher_comments.json', 'w') as f:
            json.dump(self.teachers_comments, f, indent=4)

    def show_login_page(self):
        """Display an improved login/landing page with role cards."""
        self.clear_window()

        # Gradient header using Canvas
        header_h = 160
        header = tk.Canvas(self.root, height=header_h, highlightthickness=0, bd=0, bg=self.surface_color)
        header.pack(fill='x', side='top')

        start_color = (37, 99, 235)   # blue-600
        end_color = (139, 92, 246)    # violet-500
        steps = max(1, header_h)
        for i in range(steps):
            r = int(start_color[0] + (end_color[0] - start_color[0]) * (i / steps))
            g = int(start_color[1] + (end_color[1] - start_color[1]) * (i / steps))
            b = int(start_color[2] + (end_color[2] - start_color[2]) * (i / steps))
            header.create_rectangle(0, i, self.root.winfo_screenwidth(), i + 1, outline="", fill=f'#{r:02x}{g:02x}{b:02x}')

        header.create_text(24, 56, anchor='w', text="Student Performance Analyzer", fill='white', font=self.h1_font)
        header.create_text(24, 100, anchor='w', text="Welcome! Choose your role to continue.", fill='#e5e7eb', font=('Segoe UI', 12))

        # Content surface
        surface = ttk.Frame(self.root, style='Surface.TFrame', padding=(24, 24, 24, 24))
        surface.pack(fill='both', expand=True)

        # Cards container
        cards = ttk.Frame(surface, style='Surface.TFrame')
        cards.pack(pady=16, fill='x', expand=True)

        # Role cards
        self._make_role_card(cards,
                             title="Student",
                             desc="Upload your marks CSV, view analysis, leaderboard, and export PDF.",
                             emoji="🎓",
                             button_text="Enter as Student",
                             style='Primary.TButton',
                             command=lambda: self.login_as('student')).grid(row=0, column=0, padx=12, pady=12, sticky='nsew')

        self._make_role_card(cards,
                             title="Teacher",
                             desc="Upload class CSV, view all reports, and manage comments.",
                             emoji="👩‍🏫",
                             button_text="Enter as Teacher",
                             style='Danger.TButton',
                             command=lambda: self.login_as('teacher')).grid(row=0, column=1, padx=12, pady=12, sticky='nsew')

        cards.grid_columnconfigure(0, weight=1)
        cards.grid_columnconfigure(1, weight=1)

        # Footer
        footer = ttk.Frame(surface, style='Surface.TFrame')
        footer.pack(side='bottom', fill='x', pady=(12, 0))
        ttk.Label(footer, text="Tip: You can save teacher comments; they appear in exported reports.",
                  font=('Segoe UI', 9), foreground='#9ca3af').pack()

        # Keyboard shortcuts
        self.root.bind('<Control-s>', lambda e: self.login_as('student'))
        self.root.bind('<Control-t>', lambda e: self.login_as('teacher'))

    def _make_role_card(self, parent, title, desc, emoji, button_text, style, command):
        card = ttk.Frame(parent, style='Card.TFrame', padding=20)

        ttk.Style().configure('CardHover.TFrame', background='#0d1b2a')

        def on_enter(_):
            card['padding'] = (20, 22, 20, 22)
            card.configure(style='CardHover.TFrame')

        def on_leave(_):
            card['padding'] = (20, 20, 20, 20)
            card.configure(style='Card.TFrame')

        title_lbl = ttk.Label(card, text=f"{emoji}  {title}", font=self.h2_font, foreground='white')
        desc_lbl = ttk.Label(card, text=desc, wraplength=280, foreground='#cbd5e1')
        btn = ttk.Button(card, text=button_text, style=style, command=command)

        title_lbl.pack(anchor='w')
        desc_lbl.pack(anchor='w', pady=(8, 16))
        btn.pack(anchor='w')

        card.bind("<Enter>", on_enter)
        card.bind("<Leave>", on_leave)
        for child in (title_lbl, desc_lbl):
            child.bind("<Enter>", on_enter)
            child.bind("<Leave>", on_leave)

        return card

    def login_as(self, user_type):
        """Handle login based on user type"""
        self.user_type = user_type
        if user_type == 'student':
            self.show_student_dashboard()
        else:
            self.show_teacher_dashboard()

    def show_student_dashboard(self):
        """Display student dashboard"""
        self.clear_window()

        # Header
        header = tk.Frame(self.root, bg=self.primary_color, height=80)
        header.pack(fill='x')
        title = tk.Label(header, text="Student Dashboard",
                         font=('Segoe UI', 20, 'bold'), bg=self.primary_color, fg='white')
        title.pack(pady=20)

        # Main content
        content = ttk.Frame(self.root, style='Surface.TFrame', padding=(24, 24, 24, 24))
        content.pack(fill='both', expand=True)

        upload_btn = ttk.Button(content, text="📁 Upload Marks CSV",
                                style='Success.TButton', command=self.upload_csv)
        upload_btn.pack(pady=10)

        analysis_btn = ttk.Button(content, text="📊 View Performance Analysis",
                                  style='Primary.TButton', command=self.show_analysis, state='disabled')
        analysis_btn.pack(pady=10)
        self.analysis_btn = analysis_btn

        leader_btn = ttk.Button(content, text="🏆 View Leaderboard",
                                style='Warn.TButton', command=self.show_leaderboard, state='disabled')
        leader_btn.pack(pady=10)
        self.leader_btn = leader_btn

        report_btn = ttk.Button(content, text="📄 Generate PDF Report",
                                style='Danger.TButton', command=self.generate_pdf_report, state='disabled')
        report_btn.pack(pady=10)
        self.report_btn = report_btn

        logout_btn = ttk.Button(content, text="← Logout", command=self.show_login_page)
        logout_btn.pack(pady=20)

    def show_teacher_dashboard(self):
        """Display teacher dashboard"""
        self.clear_window()

        # Header
        header = tk.Frame(self.root, bg=self.danger_color, height=80)
        header.pack(fill='x')
        title = tk.Label(header, text="Teacher Dashboard",
                         font=('Segoe UI', 20, 'bold'), bg=self.danger_color, fg='white')
        title.pack(pady=20)

        # Main content
        content = ttk.Frame(self.root, style='Surface.TFrame', padding=(24, 24, 24, 24))
        content.pack(fill='both', expand=True)

        upload_btn = ttk.Button(content, text="📁 Upload Class Marks CSV",
                                style='Success.TButton', command=self.upload_csv)
        upload_btn.pack(pady=10)

        reports_btn = ttk.Button(content, text="📊 View All Students Reports",
                                 style='Primary.TButton', command=self.show_all_reports, state='disabled')
        reports_btn.pack(pady=10)
        self.reports_btn = reports_btn

        comment_btn = ttk.Button(content, text="💬 Add/View Comments",
                                 style='Warn.TButton', command=self.manage_comments, state='disabled')
        comment_btn.pack(pady=10)
        self.comment_btn = comment_btn

        logout_btn = ttk.Button(content, text="← Logout", command=self.show_login_page)
        logout_btn.pack(pady=20)

    def upload_csv(self):
        """Handle CSV file upload"""
        file_path = filedialog.askopenfilename(
            title="Select Marks CSV File",
            filetypes=[("CSV files", ".csv"), ("All files", ".*")]
        )

        if file_path:
            try:
                self.students_data = pd.read_csv(file_path)

                # Clean column names (remove whitespace)
                self.students_data.columns = self.students_data.columns.str.strip()

                # Validate required columns
                required_cols = ['Student_Name', 'Roll_No']
                if not all(col in self.students_data.columns for col in required_cols):
                    messagebox.showerror("Error", "CSV must contain 'Student_Name' and 'Roll_No' columns!")
                    return

                messagebox.showinfo("Success", f"Loaded data for {len(self.students_data)} students!")

                # Enable buttons based on user type
                if self.user_type == 'student':
                    self.analysis_btn.config(state='normal')
                    self.leader_btn.config(state='normal')
                    self.report_btn.config(state='normal')
                else:
                    self.reports_btn.config(state='normal')
                    self.comment_btn.config(state='normal')

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load CSV: {str(e)}")

    def get_subject_columns(self):
        """Extract subject columns (MSE1 and MSE2)"""
        cols = self.students_data.columns.tolist()
        mse1_cols = [col for col in cols if 'MSE1' in col or 'MSE_1' in col or 'mse1' in col]
        mse2_cols = [col for col in cols if 'MSE2' in col or 'MSE_2' in col or 'mse2' in col]
        return mse1_cols, mse2_cols

    def analyze_student_performance(self, student_data):
        """Analyze individual student performance"""
        mse1_cols, mse2_cols = self.get_subject_columns()

        if not mse1_cols or not mse2_cols:
            return None

        # Calculate subject-wise performance
        subjects = []
        for mse1_col in mse1_cols:
            subject_name = mse1_col.replace('_MSE1', '').replace('_MSE_1', '').replace('_mse1', '')
            subjects.append(subject_name)

        analysis = {
            'subjects': [],
            'mse1_marks': [],
            'mse2_marks': [],
            'avg_marks': [],
            'total_mse1': 0,
            'total_mse2': 0,
            'overall_avg': 0
        }

        for i, subject in enumerate(subjects):
            if i < len(mse1_cols) and i < len(mse2_cols):
                mse1 = student_data[mse1_cols[i]]
                mse2 = student_data[mse2_cols[i]]
                avg = (mse1 + mse2) / 2

                analysis['subjects'].append(subject)
                analysis['mse1_marks'].append(mse1)
                analysis['mse2_marks'].append(mse2)
                analysis['avg_marks'].append(avg)

                analysis['total_mse1'] += mse1
                analysis['total_mse2'] += mse2

        if analysis['subjects']:
            analysis['overall_avg'] = (analysis['total_mse1'] + analysis['total_mse2']) / (
                        2 * len(analysis['subjects']))

        # Identify weak and strong subjects
        if analysis['avg_marks']:
            avg_threshold = np.mean(analysis['avg_marks'])
            analysis['weak_subjects'] = [s for s, m in zip(analysis['subjects'], analysis['avg_marks']) if
                                         m < avg_threshold]
            analysis['strong_subjects'] = [s for s, m in zip(analysis['subjects'], analysis['avg_marks']) if
                                           m >= avg_threshold]

        return analysis

    def predict_ese_marks(self, student_data):
        """Predict ESE marks using linear regression"""
        mse1_cols, mse2_cols = self.get_subject_columns()

        predictions = {}

        for i, subject in enumerate(mse1_cols):
            subject_name = subject.replace('_MSE1', '').replace('_MSE_1', '').replace('_mse1', '')

            if i < len(mse2_cols):
                mse1 = student_data[mse1_cols[i]]
                mse2 = student_data[mse2_cols[i]]

                # Simple linear regression: ESE = trend across exams 1,2 -> predict 3
                X = np.array([[1, mse1], [2, mse2]])
                y = np.array([mse1, mse2])

                model = LinearRegression()
                model.fit(X[:, 0].reshape(-1, 1), y)

                ese_pred = model.predict([[3]])[0]
                ese_pred = max(0, min(100, ese_pred))

                predictions[subject_name] = {
                    'mse1': mse1,
                    'mse2': mse2,
                    'ese_predicted': round(ese_pred, 2)
                }

        return predictions

    def show_analysis(self):
        """Show student analysis window"""
        if self.students_data is None:
            messagebox.showwarning("Warning", "Please upload CSV file first!")
            return

        # Create selection window
        select_window = tk.Toplevel(self.root)
        select_window.title("Select Student")
        select_window.geometry("400x500")
        select_window.configure(bg='#ecf0f1')

        tk.Label(select_window, text="Select a Student:",
                 font=('Segoe UI', 14, 'bold'), bg='#ecf0f1').pack(pady=10)

        # Listbox with scrollbar
        frame = tk.Frame(select_window, bg='#ecf0f1')
        frame.pack(padx=20, pady=10, fill='both', expand=True)

        scrollbar = tk.Scrollbar(frame)
        scrollbar.pack(side='right', fill='y')

        listbox = tk.Listbox(frame, yscrollcommand=scrollbar.set,
                             font=('Segoe UI', 11), height=15)
        listbox.pack(side='left', fill='both', expand=True)
        scrollbar.config(command=listbox.yview)

        # Populate student list
        for idx, row in self.students_data.iterrows():
            listbox.insert(tk.END, f"{row['Roll_No']} - {row['Student_Name']}")

        def show_selected():
            selection = listbox.curselection()
            if selection:
                idx_local = selection[0]
                student_data = self.students_data.iloc[idx_local]
                select_window.destroy()
                self.display_student_analysis(student_data)

        ttk.Button(select_window, text="View Analysis", style='Primary.TButton',
                   command=show_selected).pack(pady=10)

    def display_student_analysis(self, student_data):
        """Display detailed analysis for a student"""
        analysis = self.analyze_student_performance(student_data)
        predictions = self.predict_ese_marks(student_data)

        if not analysis:
            messagebox.showerror("Error", "Unable to analyze data!")
            return

        # Create analysis window
        analysis_window = tk.Toplevel(self.root)
        analysis_window.title(f"Analysis - {student_data['Student_Name']}")
        analysis_window.geometry("1000x700")
        analysis_window.configure(bg='white')

        # Header
        header = tk.Frame(analysis_window, bg=self.primary_color, height=60)
        header.pack(fill='x')

        tk.Label(header, text=f"Performance Analysis - {student_data['Student_Name']}",
                 font=('Segoe UI', 18, 'bold'), bg=self.primary_color, fg='white').pack(pady=15)

        # Create notebook for tabs
        notebook = ttk.Notebook(analysis_window)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)

        # Tab 1: Performance Overview
        overview_frame = tk.Frame(notebook, bg='white')
        notebook.add(overview_frame, text='Performance Overview')

        # Create canvas with scrollbar
        canvas = tk.Canvas(overview_frame, bg='white')
        scrollbar = tk.Scrollbar(overview_frame, orient='vertical', command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='white')

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)

        # Performance Summary
        summary_frame = tk.LabelFrame(scrollable_frame, text="Performance Summary",
                                      font=('Segoe UI', 12, 'bold'), bg='white', padx=20, pady=10)
        summary_frame.pack(fill='x', padx=20, pady=10)

        tk.Label(summary_frame, text=f"Overall Average: {analysis['overall_avg']:.2f}%",
                 font=('Segoe UI', 11), bg='white').pack(anchor='w', pady=5)
        tk.Label(summary_frame, text=f"MSE-1 Total: {analysis['total_mse1']:.2f}",
                 font=('Segoe UI', 11), bg='white').pack(anchor='w', pady=5)
        tk.Label(summary_frame, text=f"MSE-2 Total: {analysis['total_mse2']:.2f}",
                 font=('Segoe UI', 11), bg='white').pack(anchor='w', pady=5)

        # Strong Subjects
        strong_frame = tk.LabelFrame(scrollable_frame, text="💪 Strong Subjects",
                                     font=('Segoe UI', 12, 'bold'), bg='white', fg='green', padx=20, pady=10)
        strong_frame.pack(fill='x', padx=20, pady=10)

        if analysis.get('strong_subjects'):
            for subj in analysis['strong_subjects']:
                tk.Label(strong_frame, text=f"✓ {subj}",
                         font=('Segoe UI', 11), bg='white', fg='green').pack(anchor='w', pady=2)
        else:
            tk.Label(strong_frame, text="No strong subjects identified",
                     font=('Segoe UI', 11), bg='white').pack(anchor='w')

        # Weak Subjects
        weak_frame = tk.LabelFrame(scrollable_frame, text="⚠ Subjects Need Improvement",
                                   font=('Segoe UI', 12, 'bold'), bg='white', fg='red', padx=20, pady=10)
        weak_frame.pack(fill='x', padx=20, pady=10)

        if analysis.get('weak_subjects'):
            for subj in analysis['weak_subjects']:
                tk.Label(weak_frame, text=f"✗ {subj}",
                         font=('Segoe UI', 11), bg='white', fg='red').pack(anchor='w', pady=2)

            tk.Label(weak_frame, text="\n📌 Recommendations:",
                     font=('Segoe UI', 11, 'bold'), bg='white').pack(anchor='w', pady=5)
            tk.Label(weak_frame, text="• Focus more time on weak subjects during study sessions",
                     font=('Segoe UI', 10), bg='white').pack(anchor='w', padx=20)
            tk.Label(weak_frame, text="• Seek help from teachers or peers for these subjects",
                     font=('Segoe UI', 10), bg='white').pack(anchor='w', padx=20)
            tk.Label(weak_frame, text="• Practice previous year questions and solve more problems",
                     font=('Segoe UI', 10), bg='white').pack(anchor='w', padx=20)
        else:
            tk.Label(weak_frame, text="Great job! Keep up the good work!",
                     font=('Segoe UI', 11), bg='white', fg='green').pack(anchor='w')

        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        # Tab 2: Graphs & Predictions
        graph_frame = tk.Frame(notebook, bg='white')
        notebook.add(graph_frame, text='Graphs & ESE Prediction')

        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        fig.patch.set_facecolor('white')

        # Graph 1: MSE Comparison
        x = np.arange(len(analysis['subjects']))
        width = 0.35
        ax1.bar(x - width / 2, analysis['mse1_marks'], width, label='MSE-1', color='#3498db')
        ax1.bar(x + width / 2, analysis['mse2_marks'], width, label='MSE-2', color='#e74c3c')
        ax1.set_xlabel('Subjects', fontweight='bold')
        ax1.set_ylabel('Marks', fontweight='bold')
        ax1.set_title('MSE-1 vs MSE-2 Performance', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(analysis['subjects'], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # Graph 2: Average Performance
        ax2.plot(analysis['subjects'], analysis['avg_marks'], marker='o',
                 linewidth=2, markersize=8, color='#9b59b6')
        ax2.set_xlabel('Subjects', fontweight='bold')
        ax2.set_ylabel('Average Marks', fontweight='bold')
        ax2.set_title('Average Performance Across Subjects', fontweight='bold')
        ax2.set_xticklabels(analysis['subjects'], rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)

        # Graph 3: ESE Prediction
        pred_subjects = list(predictions.keys())
        mse1_marks = [predictions[s]['mse1'] for s in pred_subjects]
        mse2_marks = [predictions[s]['mse2'] for s in pred_subjects]
        ese_marks = [predictions[s]['ese_predicted'] for s in pred_subjects]

        ax3.plot(['MSE-1', 'MSE-2', 'ESE'],
                 [np.mean(mse1_marks), np.mean(mse2_marks), np.mean(ese_marks)],
                 marker='o', linewidth=3, markersize=10, color='#2ecc71')
        ax3.set_ylabel('Average Marks', fontweight='bold')
        ax3.set_title('Predicted ESE Performance', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.axvline(x=1.5, color='red', linestyle='--', alpha=0.5, label='Prediction')
        ax3.legend()

        # Graph 4: Subject-wise ESE Prediction
        width = 0.25
        x_pred = np.arange(len(pred_subjects))
        ax4.bar(x_pred - width, mse1_marks, width, label='MSE-1', color='#3498db')
        ax4.bar(x_pred, mse2_marks, width, label='MSE-2', color='#e74c3c')
        ax4.bar(x_pred + width, ese_marks, width, label='ESE (Predicted)', color='#2ecc71')
        ax4.set_xlabel('Subjects', fontweight='bold')
        ax4.set_ylabel('Marks', fontweight='bold')
        ax4.set_title('Subject-wise ESE Prediction', fontweight='bold')
        ax4.set_xticks(x_pred)
        ax4.set_xticklabels(pred_subjects, rotation=45, ha='right')
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        # Embed plot in tkinter
        canvas_plot = FigureCanvasTkAgg(fig, master=graph_frame)
        canvas_plot.draw()
        canvas_plot.get_tk_widget().pack(fill='both', expand=True)

        # Store for PDF generation
        self.current_analysis = {
            'student_data': student_data,
            'analysis': analysis,
            'predictions': predictions,
            'figure': fig
        }

    def show_leaderboard(self):
        """Display class leaderboard"""
        if self.students_data is None:
            messagebox.showwarning("Warning", "Please upload CSV file first!")
            return

        # Calculate total marks for each student
        mse1_cols, mse2_cols = self.get_subject_columns()

        leaderboard_data = []
        for idx, row in self.students_data.iterrows():
            total = 0
            count = 0
            for col in mse1_cols + mse2_cols:
                if col in row:
                    total += row[col]
                    count += 1

            avg = total / count if count > 0 else 0
            leaderboard_data.append({
                'Roll_No': row['Roll_No'],
                'Name': row['Student_Name'],
                'Total': total,
                'Average': avg
            })

        # Sort by average (descending)
        leaderboard_data.sort(key=lambda x: x['Average'], reverse=True)

        # Create leaderboard window
        leader_window = tk.Toplevel(self.root)
        leader_window.title("Class Leaderboard")
        leader_window.geometry("700x600")
        leader_window.configure(bg='#ecf0f1')

        # Header
        header = tk.Frame(leader_window, bg=self.warning_color, height=80)
        header.pack(fill='x')

        tk.Label(header, text="🏆 Class Leaderboard",
                 font=('Segoe UI', 20, 'bold'), bg=self.warning_color, fg='white').pack(pady=20)

        # Top 3 Students
        top3_frame = tk.Frame(leader_window, bg='white', relief='raised', bd=2)
        top3_frame.pack(fill='x', padx=20, pady=10)

        medals = ['🥇', '🥈', '🥉']
        colors_medal = ['#FFD700', '#C0C0C0', '#CD7F32']

        for i, student in enumerate(leaderboard_data[:3]):
            frame = tk.Frame(top3_frame, bg=colors_medal[i], relief='ridge', bd=3)
            frame.pack(fill='x', padx=10, pady=5)

            tk.Label(frame, text=f"{medals[i]} Rank {i + 1}",
                     font=('Segoe UI', 12, 'bold'), bg=colors_medal[i]).pack(side='left', padx=10)
            tk.Label(frame, text=f"{student['Name']} ({student['Roll_No']})",
                     font=('Segoe UI', 11), bg=colors_medal[i]).pack(side='left', padx=10)
            tk.Label(frame, text=f"Avg: {student['Average']:.2f}%",
                     font=('Segoe UI', 11, 'bold'), bg=colors_medal[i]).pack(side='right', padx=10)

        # Full Leaderboard Table
        table_frame = tk.Frame(leader_window, bg='white')
        table_frame.pack(fill='both', expand=True, padx=20, pady=10)

        # Scrollbar
        scrollbar = tk.Scrollbar(table_frame)
        scrollbar.pack(side='right', fill='y')

        # Treeview for table
        columns = ('Rank', 'Roll No', 'Name', 'Total Marks', 'Average')
        tree = ttk.Treeview(table_frame, columns=columns, show='headings',
                            yscrollcommand=scrollbar.set, height=15)

        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=120, anchor='center')

        scrollbar.config(command=tree.yview)

        # Populate table
        for i, student in enumerate(leaderboard_data, 1):
            tree.insert('', 'end', values=(
                i,
                student['Roll_No'],
                student['Name'],
                f"{student['Total']:.2f}",
                f"{student['Average']:.2f}%"
            ))

        tree.pack(fill='both', expand=True)

    def generate_pdf_report(self):
        """Generate comprehensive PDF report"""
        if not hasattr(self, 'current_analysis'):
            messagebox.showwarning("Warning", "Please view student analysis first!")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")],
            initialfile=f"Report_{self.current_analysis['student_data']['Roll_No']}.pdf"
        )

        if not file_path:
            return

        try:
            # Create PDF
            doc = SimpleDocTemplate(file_path, pagesize=letter)
            story = []
            styles = getSampleStyleSheet()

            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#2c3e50'),
                spaceAfter=30,
                alignment=1
            )

            story.append(Paragraph("STUDENT PERFORMANCE REPORT", title_style))
            story.append(Spacer(1, 0.3 * inch))

            # Student Info
            student = self.current_analysis['student_data']
            info_data = [
                ['Student Name:', student['Student_Name']],
                ['Roll Number:', student['Roll_No']],
                ['Report Date:', datetime.now().strftime('%B %d, %Y')],
            ]

            info_table = Table(info_data, colWidths=[2 * inch, 4 * inch])
            info_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#3498db')),
                ('TEXTCOLOR', (0, 0), (0, -1), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 11),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(info_table)
            story.append(Spacer(1, 0.4 * inch))

            # Performance Summary
            analysis = self.current_analysis['analysis']
            story.append(Paragraph("PERFORMANCE SUMMARY", styles['Heading2']))
            story.append(Spacer(1, 0.2 * inch))

            summary_data = [
                ['Metric', 'Value'],
                ['Overall Average', f"{analysis['overall_avg']:.2f}%"],
                ['MSE-1 Total', f"{analysis['total_mse1']:.2f}"],
                ['MSE-2 Total', f"{analysis['total_mse2']:.2f}"],
                ['Number of Subjects', str(len(analysis['subjects']))]
            ]

            summary_table = Table(summary_data, colWidths=[3 * inch, 2 * inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#ecf0f1')])
            ]))
            story.append(summary_table)
            story.append(Spacer(1, 0.4 * inch))

            # Subject-wise Performance
            story.append(Paragraph("SUBJECT-WISE PERFORMANCE", styles['Heading2']))
            story.append(Spacer(1, 0.2 * inch))

            subject_data = [['Subject', 'MSE-1', 'MSE-2', 'Average']]
            for i, subj in enumerate(analysis['subjects']):
                subject_data.append([
                    subj,
                    f"{analysis['mse1_marks'][i]:.2f}",
                    f"{analysis['mse2_marks'][i]:.2f}",
                    f"{analysis['avg_marks'][i]:.2f}"
                ])

            subject_table = Table(subject_data, colWidths=[2.5 * inch, 1.5 * inch, 1.5 * inch, 1.5 * inch])
            subject_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#9b59b6')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#ecf0f1')])
            ]))
            story.append(subject_table)
            story.append(Spacer(1, 0.4 * inch))

            # Strengths and Weaknesses
            story.append(Paragraph("STRENGTHS & AREAS FOR IMPROVEMENT", styles['Heading2']))
            story.append(Spacer(1, 0.2 * inch))

            strengths_text = "<b>Strong Subjects:</b><br/>"
            if analysis.get('strong_subjects'):
                for subj in analysis['strong_subjects']:
                    strengths_text += f"✓ {subj}<br/>"
            else:
                strengths_text += "None identified<br/>"

            story.append(Paragraph(strengths_text, styles['Normal']))
            story.append(Spacer(1, 0.2 * inch))

            weakness_text = "<b>Subjects Needing Improvement:</b><br/>"
            if analysis.get('weak_subjects'):
                for subj in analysis['weak_subjects']:
                    weakness_text += f"✗ {subj}<br/>"
            else:
                weakness_text += "None - Excellent performance across all subjects!<br/>"

            story.append(Paragraph(weakness_text, styles['Normal']))
            story.append(Spacer(1, 0.3 * inch))

            # Recommendations
            story.append(Paragraph("RECOMMENDATIONS", styles['Heading2']))
            story.append(Spacer(1, 0.2 * inch))

            recommendations = """
            • Maintain consistent study schedule for all subjects<br/>
            • Focus additional time on subjects with lower performance<br/>
            • Seek clarification from teachers for difficult concepts<br/>
            • Practice previous exam papers and sample questions<br/>
            • Form study groups with peers for collaborative learning<br/>
            • Take regular breaks and maintain good health habits
            """
            story.append(Paragraph(recommendations, styles['Normal']))
            story.append(PageBreak())

            # ESE Predictions
            story.append(Paragraph("ESE MARKS PREDICTION", styles['Heading2']))
            story.append(Spacer(1, 0.2 * inch))

            predictions = self.current_analysis['predictions']
            pred_data = [['Subject', 'MSE-1', 'MSE-2', 'Predicted ESE']]
            for subj, pred in predictions.items():
                pred_data.append([
                    subj,
                    f"{pred['mse1']:.2f}",
                    f"{pred['mse2']:.2f}",
                    f"{pred['ese_predicted']:.2f}"
                ])

            pred_table = Table(pred_data, colWidths=[2.5 * inch, 1.5 * inch, 1.5 * inch, 1.8 * inch])
            pred_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2ecc71')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#ecf0f1')])
            ]))
            story.append(pred_table)
            story.append(Spacer(1, 0.3 * inch))

            note_text = """
            <b>Note:</b> ESE predictions are based on linear regression analysis of MSE-1 and MSE-2 performance. 
            Actual results may vary based on preparation and exam difficulty.
            """
            story.append(Paragraph(note_text, styles['Normal']))
            story.append(Spacer(1, 0.4 * inch))

            # Save figure as image and add to PDF
            img_buffer = io.BytesIO()
            self.current_analysis['figure'].savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)

            story.append(Paragraph("PERFORMANCE GRAPHS", styles['Heading2']))
            story.append(Spacer(1, 0.2 * inch))
            story.append(Image(img_buffer, width=7 * inch, height=5.5 * inch))

            # Teacher Comments (if available)
            roll_no = str(student['Roll_No'])
            if roll_no in self.teachers_comments:
                story.append(PageBreak())
                story.append(Paragraph("TEACHER'S COMMENTS", styles['Heading2']))
                story.append(Spacer(1, 0.2 * inch))

                comment_text = self.teachers_comments[roll_no]
                story.append(Paragraph(comment_text, styles['Normal']))

            # Build PDF
            doc.build(story)
            messagebox.showinfo("Success", f"PDF report generated successfully!\n\nSaved to: {file_path}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate PDF: {str(e)}")

    def show_all_reports(self):
        """Show all students' reports for teacher"""
        if self.students_data is None:
            messagebox.showwarning("Warning", "Please upload CSV file first!")
            return

        # Create reports window
        reports_window = tk.Toplevel(self.root)
        reports_window.title("All Students Reports")
        reports_window.geometry("900x600")
        reports_window.configure(bg='#ecf0f1')

        # Header
        header = tk.Frame(reports_window, bg=self.danger_color, height=60)
        header.pack(fill='x')

        tk.Label(header, text="All Students Performance Reports",
                 font=('Segoe UI', 18, 'bold'), bg=self.danger_color, fg='white').pack(pady=15)

        # Create table
        table_frame = tk.Frame(reports_window, bg='white')
        table_frame.pack(fill='both', expand=True, padx=20, pady=20)

        # Scrollbars
        v_scrollbar = tk.Scrollbar(table_frame)
        v_scrollbar.pack(side='right', fill='y')

        h_scrollbar = tk.Scrollbar(table_frame, orient='horizontal')
        h_scrollbar.pack(side='bottom', fill='x')

        # Treeview
        columns = ('Roll No', 'Name', 'MSE-1 Avg', 'MSE-2 Avg', 'Overall Avg', 'Status')
        tree = ttk.Treeview(table_frame, columns=columns, show='headings',
                            yscrollcommand=v_scrollbar.set,
                            xscrollcommand=h_scrollbar.set)

        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=140, anchor='center')

        v_scrollbar.config(command=tree.yview)
        h_scrollbar.config(command=tree.xview)

        # Populate data
        mse1_cols, mse2_cols = self.get_subject_columns()

        for idx, row in self.students_data.iterrows():
            mse1_vals = [row[col] for col in mse1_cols if col in row]
            mse2_vals = [row[col] for col in mse2_cols if col in row]
            mse1_avg = float(np.mean(mse1_vals)) if len(mse1_vals) > 0 else 0.0
            mse2_avg = float(np.mean(mse2_vals)) if len(mse2_vals) > 0 else 0.0
            overall_avg = (mse1_avg + mse2_avg) / 2 if (mse1_vals or mse2_vals) else 0.0

            status = "Excellent" if overall_avg >= 80 else "Good" if overall_avg >= 60 else "Needs Improvement"

            tree.insert('', 'end', values=(
                row['Roll_No'],
                row['Student_Name'],
                f"{mse1_avg:.2f}",
                f"{mse2_avg:.2f}",
                f"{overall_avg:.2f}",
                status
            ))

        tree.pack(fill='both', expand=True)

        # View detailed analysis button
        def view_details():
            selection = tree.selection()
            if selection:
                item = tree.item(selection[0])
                roll_no = item['values'][0]
                student_data = self.students_data[self.students_data['Roll_No'] == roll_no].iloc[0]
                self.display_student_analysis(student_data)

        btn_frame = tk.Frame(reports_window, bg='#ecf0f1')
        btn_frame.pack(pady=10)

        ttk.Button(btn_frame, text="View Detailed Analysis",
                   style='Primary.TButton', command=view_details).pack()

    def manage_comments(self):
        """Teacher interface to add/view comments"""
        if self.students_data is None:
            messagebox.showwarning("Warning", "Please upload CSV file first!")
            return

        # Create comment window
        comment_window = tk.Toplevel(self.root)
        comment_window.title("Manage Student Comments")
        comment_window.geometry("700x600")
        comment_window.configure(bg='#ecf0f1')

        # Header
        header = tk.Frame(comment_window, bg=self.primary_color, height=60)
        header.pack(fill='x')

        tk.Label(header, text="Add/View Student Comments",
                 font=('Segoe UI', 18, 'bold'), bg=self.primary_color, fg='white').pack(pady=15)

        # Student selection
        select_frame = tk.Frame(comment_window, bg='#ecf0f1')
        select_frame.pack(fill='x', padx=20, pady=10)

        tk.Label(select_frame, text="Select Student:",
                 font=('Segoe UI', 12, 'bold'), bg='#ecf0f1').pack(side='left', padx=10)

        student_var = tk.StringVar()
        student_list = [f"{row['Roll_No']} - {row['Student_Name']}"
                        for _, row in self.students_data.iterrows()]

        student_combo = ttk.Combobox(select_frame, textvariable=student_var,
                                     values=student_list, width=40, state='readonly')
        student_combo.pack(side='left', padx=10)

        # Comment text area
        comment_frame = tk.LabelFrame(comment_window, text="Comment",
                                      font=('Segoe UI', 12, 'bold'), bg='white', padx=10, pady=10)
        comment_frame.pack(fill='both', expand=True, padx=20, pady=10)

        comment_text = tk.Text(comment_frame, font=('Segoe UI', 11), wrap='word', height=15)
        comment_text.pack(fill='both', expand=True)

        # Scrollbar for text
        scrollbar = tk.Scrollbar(comment_text)
        scrollbar.pack(side='right', fill='y')
        comment_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=comment_text.yview)

        def load_comment():
            """Load existing comment for selected student"""
            selection = student_combo.get()
            if selection:
                roll_no_local = selection.split(' - ')[0]
                if roll_no_local in self.teachers_comments:
                    comment_text.delete('1.0', tk.END)
                    comment_text.insert('1.0', self.teachers_comments[roll_no_local])
                else:
                    comment_text.delete('1.0', tk.END)

        student_combo.bind('<<ComboboxSelected>>', lambda e: load_comment())

        def save_comment():
            """Save comment for selected student"""
            selection = student_combo.get()
            if not selection:
                messagebox.showwarning("Warning", "Please select a student!")
                return

            roll_no_local = selection.split(' - ')[0]
            comment = comment_text.get('1.0', tk.END).strip()

            if comment:
                self.teachers_comments[roll_no_local] = comment
                self.save_comments()
                messagebox.showinfo("Success", "Comment saved successfully!")
            else:
                messagebox.showwarning("Warning", "Please enter a comment!")

        # Buttons
        btn_frame = tk.Frame(comment_window, bg='#ecf0f1')
        btn_frame.pack(pady=10)

        ttk.Button(btn_frame, text="Save Comment",
                   style='Success.TButton', command=save_comment).pack(side='left', padx=10)

        ttk.Button(btn_frame, text="Clear",
                   style='Warn.TButton', command=lambda: comment_text.delete('1.0', tk.END)).pack(side='left', padx=10)

    def clear_window(self):
        """Clear all widgets from window"""
        for widget in self.root.winfo_children():
            widget.destroy()

    def run(self):
        """Run the application"""
        self.root.mainloop()


if _name_ == "_main_":
    app = StudentPerformanceAnalyzer()
    app.run()
