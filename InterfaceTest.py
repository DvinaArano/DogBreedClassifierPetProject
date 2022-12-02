from tkinter import *
from tkinter import messagebox as mb
from tkinter import filedialog as fd
from tkinter import StringVar
from tkinter.ttk import Progressbar
from PIL import ImageTk, Image
from dog_app_2 import kickstart
import webbrowser

class App:
    def __init__(self):
        OptionList = ["Собака", "Кошка"]

        self.root = Tk()
        self.root.configure(bg='#fefaef')
        self.root.geometry("1000x450+300+300")
        self.root.resizable(width=True, height=True)

        self.c = Canvas(self.root, width=350, height=350, bg='white')
        self.c.grid(row=0, column=0, columnspan=5, rowspan=8,sticky=NW, pady=10, padx=10)

        self.UploadImage = Button(text="Загрузить изображение", command=self.find_file, bg="#e99674")
        self.UploadImage.grid(row=3, column=11, padx=10)

        self.StartAnalise = Button(text="Старт", command=self.start, width=15, bg="#e99674")
        self.StartAnalise.grid(row=9, column=4)

        self.AnaliseResult = Label(text="Результат:", justify=LEFT, bg="#fefaef", font=("Courier", 10))
        self.AnaliseResult.grid(row=0, column=5, columnspan=5, padx=10)

        self.Wikipage = Button(text="Википедия", command=self.open_wiki, bg="#e99674", width=15)
        self.Wikipage.grid(row=3, column=10)

        self.AdditionalInfo = Button(text="Дополнительно", command=self.open_new_window, bg="#e99674", width=15)
        self.AdditionalInfo.grid(row=4, column=10)

        self.variable = StringVar(self.root)
        self.variable.set(OptionList[0])
        self.ChooseType = OptionMenu(self.root, self.variable, *OptionList)
        self.ChooseType.config(bg="#e99674", activebackground="#e99674", width=15, justify=LEFT)
        self.ChooseType["menu"].config(bg="#e99674", activebackground="#e99674")
        self.ChooseType.grid(row=4, column=11)
        self.root.mainloop()


    def find_file(self):
        self.filedial = fd.askopenfilename()
        self.img = Image.open(self.filedial)
        self.img = self.img.resize((350, 350), Image.ANTIALIAS)
        self.img = ImageTk.PhotoImage(self.img)
        self.c.create_image(0, 0, image=self.img, anchor="nw")


    def start (self):
        try:
            self.result = self.show_result(self.filedial)
        except BaseException:
            self.errorbox =mb.showerror(title="Ошибка", message="Выберете фото для анализа")
        if self.variable.get() == "Кошка":
            mb.showinfo(title="Внимание", message="Возможен анализ только собак")
        self.AnaliseResult.configure(text="Результат:\n" + self.result)
        self.MostProbable = self.result.split(sep=":", maxsplit=1)

    def show_result(self, img_path):
        result = kickstart(img_path)
        return result

    def open_wiki(self):
        try:
            webbrowser.open("https://en.wikipedia.org/wiki/"+self.MostProbable[0])
        except BaseException:
            self.errorbox = mb.showerror(title="Ошибка", message="Неизвестная порода")



    def open_new_window(self):
        try:
            animal_name = self.MostProbable[0]
        except BaseException:
            self.errorbox = mb.showerror(title="Ошибка", message="Неизвестная порода")
        if self.MostProbable[0] == "Curly-coated retriever":
            self.newWindow = Toplevel(self.root)
            self.newWindow.title("Дополнительно")
            self.newWindow.geometry("300x500")
            Aggresion_var = DoubleVar()
            Activities_var = DoubleVar()
            Handling_var = DoubleVar()
            Friendliness_var = DoubleVar()
            Intelligent_var = DoubleVar()
            Guard_var = DoubleVar()
            Lonliness_var = DoubleVar()

            Label(self.newWindow, text="Агресивность", justify=LEFT).grid(row=0, column=0,padx= 10, pady=10)
            Agression = Progressbar(self.newWindow, variable=Aggresion_var, orient=HORIZONTAL,
                               length=280, mode='determinate').grid(row=1, column=0, padx= 10)

            Label(self.newWindow, text="Активность", justify=LEFT).grid(row=2, column=0, padx= 10, pady=10)
            Activities = Progressbar(self.newWindow, variable=Activities_var, orient=HORIZONTAL,
                                length=280, mode='determinate').grid(row=3, column=0)

            Label(self.newWindow, text="Дрессировка", justify=LEFT).grid(row=4, column=0, padx= 10, pady=10)
            Handling = Progressbar(self.newWindow,variable=Handling_var, orient=HORIZONTAL,
                                length=280, mode='determinate').grid(row=5, column=0)

            Label(self.newWindow, text="Дружелюбие", justify=LEFT).grid(row=6, column=0, padx= 10, pady=10)
            Friendliness = Progressbar(self.newWindow,variable=Friendliness_var, orient=HORIZONTAL,
                                length=280, mode='determinate').grid(row=7, column=0)

            Label(self.newWindow, text="Интелект", justify=LEFT).grid(row=8, column=0, padx= 10, pady=10)
            Intelligent = Progressbar(self.newWindow,variable=Intelligent_var, orient=HORIZONTAL,
                                length=280, mode='determinate').grid(row=9, column=0)

            Label(self.newWindow, text="Охранные качества", justify=LEFT).grid(row=10, column=0, padx= 10, pady=10)
            Guard = Progressbar(self.newWindow,variable=Guard_var, orient=HORIZONTAL,
                                length=280, mode='determinate').grid(row=11, column=0)

            Label(self.newWindow, text="Отношение к одиночеству", justify=LEFT).grid(row=12, column=0, padx= 10, pady=10)
            Lonliness = Progressbar(self.newWindow,variable=Lonliness_var, orient=HORIZONTAL,
                            length=280, mode='determinate').grid(row=13, column=0)

            Aggresion_var.set(30)
            Activities_var.set(40)
            Handling_var.set(70)
            Friendliness_var.set(70)
            Intelligent_var.set(80)
            Guard_var.set(70)
            Lonliness_var.set(10)
        else:
            mb.showinfo(title="Внимание", message="Данная порода ещё не добавлена в справочник")


        #Exit = Button(self.newWindow, text="Выход", bg="#e99674", command = self.newWindow.quit(), width = 15, height = 1).grid(row=14, column=0, padx= 10, pady=10)
        #self.newWindow.mainloop()




app =App()



