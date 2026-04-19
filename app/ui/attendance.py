from customtkinter import CTkLabel, CTkButton, CTkScrollableFrame 
import cv2 

class AttendancePage(CTkScrollableFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs, fg_color="transparent")
        
        # --- UI SETUP ---
        self.grid_columnconfigure(0, weight=1)
        self.title = CTkLabel(self, text="Attendance Page", font=("Times New Roman", 20, "bold"))
        self.title.grid(row=0, column=0, padx=10, pady=10)