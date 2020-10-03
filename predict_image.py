from tensorflow.keras.models import load_model
from tkinter import *
import tkinter as tk
import win32gui
from PIL import ImageGrab, Image
import numpy as np
import tensorflow as tf



letters = { 1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I', 10: 'J',
11: 'K', 12: 'L', 13: 'M', 14: 'N', 15: 'O', 16: 'P', 17: 'Q', 18: 'R', 19: 'S', 20: 'T',
21: 'U', 22: 'V', 23: 'W', 24: 'X', 25: 'Y', 26: 'Z', 27: '-'}

model=load_model('C:/Users/xxx/models.h5')
model_rakam=load_model('C:/Users/xxx/Desktop/mnıst/mnist_model1.h5')

def predict_digit1(img):
    #resize image to 28x28 pixels
    img = img.resize((28,28))
    #convert rgb to grayscale
    img = img.convert('L')
    img = np.array(img)
    #reshaping to support our model input and normalizing
    img = img.reshape(1,28,28,1)
    img = img/255.0
    #predicting the class
    res = model_rakam.predict([img])[0]
    return np.argmax(res), max(res)
def predict_digit2(img):
    #resize image to 28x28 pixels
    img = img.resize((28,28))
    #convert rgb to grayscale
    img = img.convert('L')
    img = np.array(img)
    #reshaping to support our model input and normalizing
    img = img.reshape(1,28,28,1)
    img = img/255.0
    #predicting the class
    res = model.predict([img])[0]
    return np.argmax(res), max(res)
class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        
        self.x = self.y = 0
        # Creating elements
        self.canvas = tk.Canvas(self, width=300, height=300, bg = "black", cursor="cross",relief=RIDGE)
        
        self.label = tk.Label(self, text="TAHMİN SONUCU", font=("Helvetica", 24))
        self.classify_btn = tk.Button(self, text = "Rakam Tahmin Et", command = self.classify_handwriting,relief=GROOVE
                                     ) 
        self.classify_btn2 = tk.Button(self, text = "Harf Tahmin Et", command = self.classify_handwriting2, relief=GROOVE
                                       )
        self.button_clear = tk.Button(self, text = "Temizle", command = self.clear_all,relief=GROOVE)
        # Grid structure
        self.canvas.grid(row=0, column=0, pady=2, sticky=W, )
        self.label.grid(row=0, column=1,pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=1, padx=0)
        self.classify_btn2.grid(row=1, column=2, pady=1, padx=0)

        self.button_clear.grid(row=1, column=0, pady=1)
        #self.canvas.bind("<Motion>", self.start_pos)
        self.canvas.bind("<B1-Motion>", self.draw_lines)
       
        
        
        
    def clear_all(self):
        self.canvas.delete("all")
    def classify_handwriting(self):
        HWND = self.canvas.winfo_id() # get the handle of the canvas
        rect = win32gui.GetWindowRect(HWND) # get the coordinate of the canvas
        im = ImageGrab.grab(rect)
   
        digit, acc = predict_digit1(im)
        self.label.configure(text= str(digit)+', '+ str(int(acc*100))+'%')
    def classify_handwriting2(self):
        HWND = self.canvas.winfo_id() # get the handle of the canvas
        rect = win32gui.GetWindowRect(HWND) # get the coordinate of the canvas
        im = ImageGrab.grab(rect)
      
        
        digit, acc = predict_digit2(im)
        self.label.configure(text= str(letters[int(digit)+1])+', '+ str(int(acc*100))+'%')

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r=8
        self.canvas.create_oval(self.x-r, self.y-r, self.x + r, self.y + r, fill='white',outline='white')
        self.label.configure('TAHMİN SONUCU:')
app = App()
app.title('Harf-Rakam-Tanıma')
app.resizable(0,0)
mainloop()
