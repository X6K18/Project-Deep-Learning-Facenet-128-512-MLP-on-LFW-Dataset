from customtkinter import CTkFrame, CTkScrollableFrame
from face_recognition import FaceRecognitionPage
from register_face import FaceRegisterPage
from attendance import AttendancePage
from face_verification import FaceVerificationPage

class MainFrame(CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs, fg_color="transparent")
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        self.current_frame = None 
    
    def show_frame(self, frame_class):
        if self.current_frame is not None:
            self.current_frame.destroy()
        
        self.current_frame = frame_class(self)
        self.current_frame.grid(row=0, column=0, sticky="nsew")