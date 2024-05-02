import numpy as np
from customtkinter import *
from tkinter import *
import Functions as F
import sounddevice as sd
from scipy.io.wavfile import write

SPEAKER = '...'
P1_name = 'Edgar'
P2_name = 'Erik'
P3_name = 'Hayk'

app = Tk()
app.title("Microphone")
app.geometry("800x500")
app.resizable(False, False)
app.configure(bg = '#363737')

Text = "Turn on mircrophone"
w = 800
h = 500

ws = app.winfo_screenwidth()
hs = app.winfo_screenheight() 

x = (ws/2) - (w/2)
y = (hs/2) - (h/2) - 90

app.geometry('%dx%d+%d+%d' % (w, h, x, y))

def toggle():
    state = btn.get()
    if state == 1:
        txt.configure(text = "Microphone is ON")
    else:
        txt.configure(text = "Microphone is OFF")

btn = CTkSwitch(master = app,
                text = "Toggle microphone",
                progress_color = '#5da668',
                fg_color= '#ff2c2c',
                switch_width = 200,
                switch_height = 100,
                font = ("Arial", 40),
                command = toggle,
                text_color = '#FFFEF2',
                )

txt = CTkLabel(master = app,
            text = "Microphone is OFF",
            font = ("Arial", 40),
            text_color= '#FFFEF2',
            )
speaker = CTkLabel(master = app,
                text = f'Now is speaking: {SPEAKER}',
                font = ("Arial", 40),
                text_color= '#FFFEF2',
                )

btn.place(relx = 0.5, rely = 0.3, anchor = CENTER)
txt.place(relx = 0.5, rely = 0.5, anchor = CENTER)
speaker.place(relx = 0.5, rely = 0.7, anchor = CENTER)

P1 = 0
P2 = 0
P3 = 0
sum_try = 0.1
flag = 0
while True:
    app.update()
    if btn.get() == 1:
        flag = 1
        if SPEAKER != '...':
            SPEAKER = '...'
            speaker.configure(text = f'Now is speaking: {SPEAKER}')
        freq = 22050
        
        duration = 0.5 # Seconds

        recording = sd.rec(int(duration * freq), 
                        samplerate=freq, channels=1)

        # Start recording
        sd.wait()

        write("Audio/recording0.wav", freq, recording)

        F.Clean("Audio/recording0.wav")
        F.Split(100)
        F.FFT() # Creating TestingData.csv
        result = F.Predict("Data/TestingData.csv", 'Model/model.pkl')
        
        dct = {-1: 'Idk', 0: P1_name, 1: P2_name, 2: P3_name}
        if result == 0:
            P1+=1
        elif result == 1:
            P2+=1
        elif result == 2:
            P3+=1
        sum_try = P1 + P2 + P3
    else:
        if flag == 1:
            print(max(P2 / sum_try, P1 / sum_try, P3 / sum_try))
            
            if max(P2 / sum_try, P1 / sum_try, P3 / sum_try) < 0.5:
                SPEAKER = 'Idk'
            elif P3 > P2 and P3 > P1:
                SPEAKER = P3_name
            elif P2 > P3 and P2 > P1:
                SPEAKER = P2_name
            elif P1 > P2 and P1 > P3:
                SPEAKER = P1_name
            flag = 0
        
        P1 = 0
        P2 = 0
        P3 = 0
        sum_try = 0.1
        
        speaker.configure(text = f'Now is speaking: {SPEAKER}')
    
    app.update()
