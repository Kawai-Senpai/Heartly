from kivy.uix.screenmanager import ScreenManager, Screen
from kivymd.app import MDApp as App
from kivymd.uix.gridlayout import MDGridLayout as GridLayout
from kivymd.uix.button import MDFillRoundFlatButton as Button
from kivymd.uix.label import MDLabel as Label
from kivymd.uix.textfield import MDTextField as TextInput
from kivy.uix.image import Image
from kivymd.uix.slider import MDSlider as Slider
from kivymd.uix.selectioncontrol import MDCheckbox
from kivy.metrics import dp
from kivymd.uix.slider import MDSlider as Slider

import random
import re
import torch
import torch.nn as nn

device= "cpu"

Q = {
    'age':[['Hi, \nwelcome to Heartily. \nLet\'s start by asking your age.', 'Hi there ! \nHow old are you?', 'Hello, \nwhat is your age?'], 
            ['Enter your age here', 'Type in your age']],
    
    'gender':[['Are you male or female?', 'Hi, please select your gender.', 'What is your gender?'], 
            ['Answer with \'male\' or \'female\'']],
    
    'cp':[['What type of chest pain do you have?', 'Could you describe your chest pain?', 'What kind of chest pain are you experiencing?'], 
            ['0 for typical angina, 1 for atypical angina, 2 for non-anginal pain, 3 for asymptomatic', 'Enter 0 for typical angina, 1 for atypical angina, 2 for non-anginal pain, or 3 for asymptomatic chest pain', 'Chest Pain Type (0-3)']],
    
    'thalach':[['What is your maximum heart rate?', 'How high can your heart rate go?', 'What is the highest heart rate you\'ve achieved?'], 
            ['Enter your maximum heart rate here', 'Maximum Heart Rate']],
    
    'exang':[['Do you experience exercise-induced angina?', 'Do you feel chest pain during exercise?', 'Have you experienced chest pain during exercise?'], 
            ['Answer with \'Yes\' or \'No\'']],
    
    'oldpeak':[['How much ST depression did you experience relative to rest?', 'What is your ST depression score?', 'Please enter your ST depression score.'], 
            ['Enter your ST depression score here']],
    
    'slope':[['What is the slope of your peak exercise ST segment?', 'How would you describe the slope of your peak exercise ST segment?', 'Please indicate the slope of your peak exercise ST segment.'], 
            ['0 for upsloping, 1 for flat, 2 for downsloping', 'Enter 0 for upsloping, 1 for flat, or 2 for downsloping', 'Peak Exercise ST Segment Slope (0-2)']],
    
    'ca':[['How many major vessels have been colored by fluoroscopy?', 'What is the count of colored major vessels by fluoroscopy?', 'Please enter the count of colored major vessels by fluoroscopy.'], 
            ['Enter the count of colored major vessels here', 'Count of colored major vessels']],
    
    'thal':[['What is your thalassemia diagnosis?', 'Have you been diagnosed with thalassemia?', 'Please indicate your thalassemia diagnosis.'], 
            ['Answer with 0 for error, 1 for fixed defect, 2 for normal, 3 for reversible defect', ]]
}
ans = {}

#Copy of the net
class neuralnet(nn.Module):

    """
    Neural network model for heart disease prediction.
    Attributes:
        fc1 (nn.Linear): Fully connected layer with input size 18 and output size 40.
        fc2 (nn.Linear): Fully connected layer with input size 40 and output size 50.
        fc3 (nn.Linear): Fully connected layer with input size 50 and output size 60.
        fc4 (nn.Linear): Fully connected layer with input size 60 and output size 70.
        fc5 (nn.Linear): Fully connected layer with input size 70 and output size 80.
        fc6 (nn.Linear): Fully connected layer with input size 80 and output size 90.
        fc7 (nn.Linear): Fully connected layer with input size 90 and output size 100.
        fc8 (nn.Linear): Fully connected layer with input size 100 and output size 110.
        fc9 (nn.Linear): Fully connected layer with input size 110 and output size 120.
        fc10 (nn.Linear): Fully connected layer with input size 120 and output size 130.
        fc11 (nn.Linear): Fully connected layer with input size 130 and output size 120.
        fc12 (nn.Linear): Fully connected layer with input size 120 and output size 110.
        fc13 (nn.Linear): Fully connected layer with input size 110 and output size 100.
        fc14 (nn.Linear): Fully connected layer with input size 100 and output size 90.
        fc15 (nn.Linear): Fully connected layer with input size 90 and output size 80.
        fc16 (nn.Linear): Fully connected layer with input size 80 and output size 70.
        fc17 (nn.Linear): Fully connected layer with input size 70 and output size 60.
        fc18 (nn.Linear): Fully connected layer with input size 60 and output size 50.
        fc19 (nn.Linear): Fully connected layer with input size 50 and output size 40.
        fc20 (nn.Linear): Fully connected layer with input size 40 and output size 30.
        fc21 (nn.Linear): Fully connected layer with input size 30 and output size 20.
        fc22 (nn.Linear): Fully connected layer with input size 20 and output size 10.
        fc23 (nn.Linear): Fully connected layer with input size 10 and output size 1.
        af (nn.GELU): Activation function GELU.
        sigmoid (nn.Sigmoid): Sigmoid activation function.
        dropout (nn.Dropout): Dropout layer with dropout rate of 0.05.
    """
    """
    Forward pass of the neural network.
    Args:
        x (torch.Tensor): Input tensor.
    Returns:
        torch.Tensor: Output tensor after passing through the network.
    """

    def __init__(self):
        super(neuralnet,self).__init__()

        self.fc1=nn.Linear(18,40)
        self.fc2=nn.Linear(40,50)
        self.fc3=nn.Linear(50,60)
        self.fc4=nn.Linear(60,70)
        self.fc5=nn.Linear(70,80)
        self.fc6=nn.Linear(80,90)
        self.fc7=nn.Linear(90,100)
        self.fc8=nn.Linear(100,110)
        self.fc9=nn.Linear(110,120)
        self.fc10=nn.Linear(120,130)
        self.fc11=nn.Linear(130,120)
        self.fc12=nn.Linear(120,110)
        self.fc13=nn.Linear(110,100)
        self.fc14=nn.Linear(100,90)
        self.fc15=nn.Linear(90,80)
        self.fc16=nn.Linear(80,70)
        self.fc17=nn.Linear(70,60)
        self.fc18=nn.Linear(60,50)
        self.fc19=nn.Linear(50,40)
        self.fc20=nn.Linear(40,30)
        self.fc21=nn.Linear(30,20)
        self.fc22=nn.Linear(20,10)
        self.fc23=nn.Linear(10,1)
        self.af=nn.GELU()
        self.sigmoid=nn.Sigmoid()
        self.dropout = nn.Dropout(0.05)

    def forward(self,x):
        
        out = self.fc1(x)
        out = self.af(out)
        out = self.fc2(out)
        out = self.af(out)
        out = self.fc3(out)

        out = self.fc4(out)
        out = self.af(out)
        out = self.fc5(out)
        out = self.af(out)
        out = self.fc6(out)
        out = self.dropout(out)
        out = self.fc7(out)
        out = self.af(out)
        out = self.fc8(out)
        out = self.af(out)
        out = self.fc9(out)
        out = self.dropout(out)
        out = self.fc10(out)
        out = self.af(out)
        out = self.fc11(out)
        out = self.af(out)
        out = self.fc12(out)
        out = self.dropout(out)
        out = self.fc13(out)
        out = self.af(out)
        out = self.fc14(out)
        out = self.af(out)
        out = self.fc15(out)
        out = self.dropout(out)
        out = self.fc16(out)
        out = self.af(out)
        out = self.fc17(out)
        out = self.af(out)
        out = self.fc18(out)
        out = self.dropout(out)
        out = self.fc19(out)
        out = self.af(out)
        out = self.fc20(out)
        out = self.af(out)
        out = self.fc21(out)
        out = self.fc22(out)
        out = self.fc23(out)

        out=self.sigmoid(out)
        return(out)

class Logo(Screen):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.next_screen = 'age'
        #Layout
        layout = GridLayout(cols=1, spacing=30)
        layout.size_hint = (0.9,0.8)
        layout.pos_hint = {"center_x":0.5, "center_y":0.5}
        layout.padding = [30,0,30,50]

        #Image
        self.image = Image(source='Heartly_LOGO.png',size_hint=(0.9, 0.9), pos_hint={"center_x": 0.5, "center_y": 0.5})
        layout.add_widget(self.image)

        #The submit button
        self.submit_button = Button(text='Go', on_press=self.next,size_hint = (0.9,0.1), pos_hint={"center_x": 0.5, "y": 0})
        layout.add_widget(self.submit_button)

        #Attach everything
        self.add_widget(layout)

    #Switch to the next screen
    def next(self,instance):

        self.manager.current = self.next_screen

class AgeScreen(Screen):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        global Q
        self.key = 'age'
        self.next_screen = 'gender'
        #Layout
        layout = GridLayout(cols=1, spacing=30)
        layout.size_hint = (0.9,0.8)
        layout.pos_hint = {"center_x":0.5, "center_y":0.5}
        layout.padding = [30,0,30,50]

        #The writing, label
        self.text = Label(text=random.choice(Q[self.key][0]))
        self.text.font_size=27
        self.text.color = (1,0.4,0.4)
        layout.add_widget(self.text)

        #The Actualinput
        self.input = TextInput(text=random.choice(Q[self.key][1]), multiline=False)
        layout.add_widget(self.input)

        #The submit button
        self.submit_button = Button(text='Next', on_press=self.next,size_hint = (0.9,0.1), pos_hint={"center_x": 0.5, "y": 0})
        
        layout.add_widget(self.submit_button)

        #Attach everything
        self.add_widget(layout)

    #Switch to the next screen
    def next(self,instance):

        global ans

        out = re.findall(r'\d+\.?\d*', self.input.text)
        if(len(out) == 0):
            out = 0
        else:
            out = out[0]

        ans[self.key] = float(out)
        print("Debug : Updated Struct -->",ans)
        self.manager.current = self.next_screen

class GenderScreen(Screen):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        global Q
        self.key = 'gender'
        self.next_screen = 'cp'
        #Layout

        self.male = 0.0
        self.female = 0.0

        layout = GridLayout(cols=1, spacing=30)
        layout.size_hint = (0.9,0.8)
        layout.pos_hint = {"center_x":0.5, "center_y":0.5}
        layout.padding = [30,0,30,50]

        #The writing, label
        self.text = Label(text=random.choice(Q[self.key][0]))
        self.text.font_size=27
        self.text.color = (1,0.4,0.4)
        layout.add_widget(self.text)

        def checkbox_callback(checkbox):
            if (checkbox.group=='Male'):
                if(checkbox.state == "down"):
                    self.male = 1.0
                else:
                    self.male = 0.0
            else:
                if(checkbox.state == "down"):
                    self.female = 1.0
                else:
                    self.female = 0.0

        option_size = 20
        option_color = (0.5,0.5,0.5)
        # The options
        male_checkbox = MDCheckbox(group="Male", size_hint=(None, None), size=(48, 48))
        male_checkbox.bind(on_release=checkbox_callback)
        layout.add_widget(male_checkbox)

        male_label = Label(text="Male", size_hint=(None, None), size=(dp(100), dp(48)), valign='middle')
        male_label.color = option_color
        male_label.font_size = option_size
        layout.add_widget(male_label)

        female_checkbox = MDCheckbox(group="Female", size_hint=(None, None), size=(48, 48))
        female_checkbox.bind(on_release=checkbox_callback)
        layout.add_widget(female_checkbox)

        female_label = Label(text="Female", size_hint=(None, None), size=(dp(100), dp(48)), valign='middle')
        female_label.color = option_color
        female_label.font_size = option_size
        layout.add_widget(female_label)

        '''#The Actualinput
        self.input = TextInput(text=random.choice(Q[self.key][1]), multiline=False)
        layout.add_widget(self.input)'''

        
        #The submit button
        self.submit_button = Button(text='Next', on_press=self.next,size_hint = (0.9,0.1), pos_hint={"center_x": 0.5, "y": 0})
        layout.add_widget(self.submit_button)

        #Attach everything
        self.add_widget(layout)

    #Switch to the next screen
    def next(self,instance):

        global ans

        ans["sex_1.0"] = self.male
        ans["sex_0.0"] = self.female

        print("Debug : Updated Struct -->",ans)
        self.manager.current = self.next_screen

class CpScreen(Screen):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        global Q
        self.key = 'cp'
        self.next_screen = 'thalach'

        self.cp_0 = 0.0 
        self.cp_1 = 0.0
        self.cp_2 = 0.0
        self.cp_3 = 0.0

        #Layout
        layout = GridLayout(cols=1, spacing=30)
        layout.size_hint = (0.9,0.8)
        layout.pos_hint = {"center_x":0.5, "center_y":0.5}
        layout.padding = [30,0,30,50]

        #The writing, label
        self.text = Label(text=random.choice(Q[self.key][0]))
        self.text.font_size=27
        self.text.color = (1,0.4,0.4)
        layout.add_widget(self.text)

        option_size = 20
        option_color = (0.5,0.5,0.5)

        def checkbox_callback(checkbox):
                if checkbox.group == "Typical angina":
                    if checkbox.state == "down":
                        self.cp_0 = 1.0
                    else:
                        self.cp_0 = 0.0
                elif checkbox.group == "Atypical angina":
                    if checkbox.state == "down":
                        self.cp_1 = 1.0
                    else:
                        self.cp_1 = 0.0
                elif checkbox.group == "Non-anginal pain":
                    if checkbox.state == "down":
                        self.cp_2 = 1.0
                    else:
                        self.cp_2 = 0.0
                else:
                    if checkbox.state == "down":
                        self.cp_3 = 1.0
                    else:
                        self.cp_3 = 0.0

        # The options
        male_checkbox = MDCheckbox(group="Typical angina", size_hint=(None, None), size=(48, 48))
        male_checkbox.bind(on_release=checkbox_callback)
        layout.add_widget(male_checkbox)

        male_label = Label(text="Typical angina", size_hint=(None, None), size=(dp(300), dp(48)), valign='middle')
        male_label.color = option_color
        male_label.font_size = option_size
        layout.add_widget(male_label)

        female_checkbox = MDCheckbox(group="Atypical angina", size_hint=(None, None), size=(48, 48))
        female_checkbox.bind(on_release=checkbox_callback)
        layout.add_widget(female_checkbox)

        female_label = Label(text="Atypical angina", size_hint=(None, None), size=(dp(300), dp(48)), valign='middle')
        female_label.color = option_color
        female_label.font_size = option_size
        layout.add_widget(female_label)

        non_anginal_checkbox = MDCheckbox(group="Non-anginal pain", size_hint=(None, None), size=(48, 48))
        non_anginal_checkbox.bind(on_release=checkbox_callback)
        layout.add_widget(non_anginal_checkbox)

        non_anginal_label = Label(text="Non-anginal pain", size_hint=(None, None), size=(dp(300), dp(48)), valign='middle')
        non_anginal_label.color = option_color
        non_anginal_label.font_size = option_size
        layout.add_widget(non_anginal_label)

        asymptomatic_checkbox = MDCheckbox(group="Asymptomatic", size_hint=(None, None), size=(48, 48))
        asymptomatic_checkbox.bind(on_release=checkbox_callback)
        layout.add_widget(asymptomatic_checkbox)

        asymptomatic_label = Label(text="Asymptomatic", size_hint=(None, None), size=(dp(300), dp(48)), valign='middle')
        asymptomatic_label.color = option_color
        asymptomatic_label.font_size = option_size
        layout.add_widget(asymptomatic_label)

        #The submit button
        self.submit_button = Button(text='Next', on_press=self.next,size_hint = (0.9,0.1), pos_hint={"center_x": 0.5, "y": 0})
        layout.add_widget(self.submit_button)

        #Attach everything
        self.add_widget(layout)

    #Switch to the next screen
    def next(self,instance):

        global ans
        #'cp_0.0', 'cp_1.0', 'cp_2.0', 'cp_3.0'

        ans["cp_0.0"] = self.cp_0
        ans["cp_1.0"] = self.cp_1
        ans["cp_2.0"] = self.cp_2
        ans["cp_3.0"] = self.cp_3

        print("Debug : Updated Struct -->",ans)
        self.manager.current = self.next_screen

class ThalachScreen(Screen):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        global Q
        self.key = 'thalach'
        self.next_screen = 'exang'
        #Layout
        layout = GridLayout(cols=1, spacing=30)
        layout.size_hint = (0.9,0.8)
        layout.pos_hint = {"center_x":0.5, "center_y":0.5}
        layout.padding = [30,0,30,50]

        self.exang = 0.0

        #The writing, label
        self.text = Label(text=random.choice(Q[self.key][0]))
        self.text.font_size=27
        self.text.color = (1,0.4,0.4)
        layout.add_widget(self.text)

        #The Actualinput
        self.input = TextInput(text=random.choice(Q[self.key][1]), multiline=False)
        layout.add_widget(self.input)


        #The submit button
        self.submit_button = Button(text='Next', on_press=self.next,size_hint = (0.9,0.1), pos_hint={"center_x": 0.5, "y": 0})
        layout.add_widget(self.submit_button)

        #Attach everything
        self.add_widget(layout)

    #Switch to the next screen
    def next(self,instance):

        global ans

        out = re.findall(r'\d+\.?\d*', self.input.text)
        if(len(out) == 0):
            out = 0
        else:
            out = out[0]

        ans[self.key] = float(out)
        print("Debug : Updated Struct -->",ans)
        self.manager.current = self.next_screen

class ExangScreen(Screen):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        global Q
        self.key = 'exang'
        self.next_screen = 'oldpeak'
        #Layout
        layout = GridLayout(cols=1, spacing=30)
        layout.size_hint = (0.9,0.8)
        layout.pos_hint = {"center_x":0.5, "center_y":0.5}
        layout.padding = [30,0,30,50]

        #The writing, label
        self.text = Label(text=random.choice(Q[self.key][0]))
        self.text.font_size=27
        self.text.color = (1,0.4,0.4)
        layout.add_widget(self.text)

        option_size = 20
        option_color = (0.5,0.5,0.5)

        self.exang = 0.0

        def checkbox_callback(checkbox):
            if checkbox.group == 'yes':
                if checkbox.state == "down":
                    self.exang = 1.0
                else:
                    self.exang = 0.0

        # The options
        yes_checkbox = MDCheckbox(group="yes", size_hint=(None, None), size=(48, 48))
        yes_checkbox.bind(on_release=checkbox_callback)
        layout.add_widget(yes_checkbox)

        yes_label = Label(text="yes", size_hint=(None, None), size=(dp(200), dp(48)), valign='middle')
        yes_label.color = option_color
        yes_label.font_size = option_size
        layout.add_widget(yes_label)

        #The submit button
        self.submit_button = Button(text='Next', on_press=self.next,size_hint = (0.9,0.1), pos_hint={"center_x": 0.5, "y": 0})
        layout.add_widget(self.submit_button)

        #Attach everything
        self.add_widget(layout)

    #Switch to the next screen
    def next(self,instance):

        global ans

        if(self.exang == 1.0):
            ans["exang_0.0"] = 0.0
            ans["exang_1.0"] = 1.0
        else:
            ans["exang_0.0"] = 1.0
            ans["exang_1.0"] = 0.0

        print("Debug : Updated Struct -->",ans)
        self.manager.current = self.next_screen

class OldpeakScreen(Screen):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        global Q
        self.key = 'oldpeak'
        self.next_screen = 'slope'
        #Layout
        layout = GridLayout(cols=1, spacing=30)
        layout.size_hint = (0.9,0.8)
        layout.pos_hint = {"center_x":0.5, "center_y":0.5}
        layout.padding = [30,0,30,50]

        #The writing, label
        self.text = Label(text=random.choice(Q[self.key][0]))
        self.text.font_size=27
        self.text.color = (1,0.4,0.4)
        layout.add_widget(self.text)

        #The Actualinput
        self.input = TextInput(text=random.choice(Q[self.key][1]), multiline=False)
        layout.add_widget(self.input)

        #The submit button
        self.submit_button = Button(text='Next', on_press=self.next,size_hint = (0.9,0.1), pos_hint={"center_x": 0.5, "y": 0})
        layout.add_widget(self.submit_button)

        #Attach everything
        self.add_widget(layout)

    #Switch to the next screen
    def next(self,instance):

        global ans

        out = re.findall(r'\d+\.?\d*', self.input.text)
        if(len(out) == 0):
            out = 0
        else:
            out = out[0]

        ans[self.key] = float(out)
        print("Debug : Updated Struct -->",ans)
        self.manager.current = self.next_screen

class SlopeScreen(Screen):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        global Q
        self.key = 'slope'
        self.next_screen = 'ca'
        #Layout
        layout = GridLayout(cols=1, spacing=30)
        layout.size_hint = (0.9,0.8)
        layout.pos_hint = {"center_x":0.5, "center_y":0.5}
        layout.padding = [30,0,30,50]

        #The writing, label
        self.text = Label(text=random.choice(Q[self.key][0]))
        self.text.font_size=27
        self.text.color = (1,0.4,0.4)
        layout.add_widget(self.text)

        option_size = 20
        option_color = (0.5, 0.5, 0.5)

        self.slope_0 = 0.0
        self.slope_1 = 0.0
        self.slope_2 = 0.0

        def checkbox_callback(checkbox):
            if checkbox.group == "Upsloping":
                if checkbox.state == "down":
                    self.slope_0 = 1.0
                else:
                    self.slope_0 = 0.0
            elif checkbox.group == "Flat":
                if checkbox.state == "down":
                    self.slope_1 = 1.0
                else:
                    self.slope_1 = 0.0
            else:
                if checkbox.state == "down":
                    self.slope_2 = 1.0
                else:
                    self.slope_2 = 0.0

        # The options
        upsloping_checkbox = MDCheckbox(group="Upsloping", size_hint=(None, None), size=(48, 48))
        upsloping_checkbox.bind(on_release=checkbox_callback)
        layout.add_widget(upsloping_checkbox)

        upsloping_label = Label(text="Upsloping", size_hint=(None, None), size=(dp(300), dp(48)), valign='middle')
        upsloping_label.color = option_color
        upsloping_label.font_size = option_size
        layout.add_widget(upsloping_label)

        flat_checkbox = MDCheckbox(group="Flat", size_hint=(None, None), size=(48, 48))
        flat_checkbox.bind(on_release=checkbox_callback)
        layout.add_widget(flat_checkbox)

        flat_label = Label(text="Flat", size_hint=(None, None), size=(dp(300), dp(48)), valign='middle')
        flat_label.color = option_color
        flat_label.font_size = option_size
        layout.add_widget(flat_label)

        downsloping_checkbox = MDCheckbox(group="Downsloping", size_hint=(None, None), size=(48, 48))
        downsloping_checkbox.bind(on_release=checkbox_callback)
        layout.add_widget(downsloping_checkbox)

        downsloping_label = Label(text="Downsloping", size_hint=(None, None), size=(dp(300), dp(48)), valign='middle')
        downsloping_label.color = option_color
        downsloping_label.font_size = option_size
        layout.add_widget(downsloping_label)


        #The submit button
        self.submit_button = Button(text='Next', on_press=self.next,size_hint = (0.9,0.1), pos_hint={"center_x": 0.5, "y": 0})
        layout.add_widget(self.submit_button)

        #Attach everything
        self.add_widget(layout)

    #Switch to the next screen
    def next(self,instance):

        global ans
        
        ans['slope_0.0'] = self.slope_0
        ans['slope_1.0'] = self.slope_1
        ans['slope_2.0'] = self.slope_2

        print("Debug : Updated Struct -->",ans)
        self.manager.current = self.next_screen

class CaScreen(Screen):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        global Q
        self.key = 'ca'
        self.next_screen = 'thal'
        #Layout
        layout = GridLayout(cols=1, spacing=30)
        layout.size_hint = (0.9,0.8)
        layout.pos_hint = {"center_x":0.5, "center_y":0.5}
        layout.padding = [30,0,30,50]

        #The writing, label
        self.text = Label(text=random.choice(Q[self.key][0]))
        self.text.font_size=27
        self.text.color = (1,0.4,0.4)
        layout.add_widget(self.text)

        self.ca = 0.0

        option_size = 20
        option_color = (0.5,0.5,0.5)

        def slider_callback(instance, value):
            selected_option = int(value)
            # Do something with the selected option
            self.ca = selected_option

        # The options
        #Turn off shadow of the slider by setting the size of the shadow to 0 syntax : shadow_size=(0, 0)
        ca_slider = Slider(min=0, max=3, value=0, step=1)
        ca_slider.bind(value=slider_callback)
        layout.add_widget(ca_slider)

        ca_label = Label(text="Number of major vessels (0-3)", size_hint=(None, None), size=(dp(300), dp(48)), valign='middle')
        ca_label.color = option_color
        ca_label.font_size = option_size
        layout.add_widget(ca_label)

        #The submit button
        self.submit_button = Button(text='Next', on_press=self.next,size_hint = (0.9,0.1), pos_hint={"center_x": 0.5, "y": 0})
        layout.add_widget(self.submit_button)

        #Attach everything
        self.add_widget(layout)

    #Switch to the next screen
    def next(self,instance):

        global ans

        ans[self.key] = self.ca
        print("Debug : Updated Struct -->",ans)
        self.manager.current = self.next_screen

class ThalScreen(Screen):

    def __init__(self,res,**kwargs):
        super().__init__(**kwargs)

        global Q
        self.res = res
        self.key = 'thal'
        self.next_screen = 'result'
        #Layout
        layout = GridLayout(cols=1, spacing=30)
        layout.size_hint = (0.9,0.8)
        layout.pos_hint = {"center_x":0.5, "center_y":0.5}
        layout.padding = [30,0,30,50]

        #The writing, label
        self.text = Label(text=random.choice(Q[self.key][0]))
        self.text.font_size=27
        self.text.color = (1,0.4,0.4)
        layout.add_widget(self.text)

        option_size = 20
        option_color = (0.5, 0.5, 0.5)

        self.thal_0 = 0.0
        self.thal_1 = 0.0
        self.thal_2 = 0.0
        self.thal_3 = 0.0

        def checkbox_callback(checkbox):
            if checkbox.group == "Unknown":
                if checkbox.state == "down":
                    self.thal_0 = 1.0
                else:
                    self.thal_0 = 0.0
            elif checkbox.group == "Fixed Defect":
                if checkbox.state == "down":
                    self.thal_1 = 1.0
                else:
                    self.thal_1 = 0.0
            elif checkbox.group == "Normal":
                if checkbox.state == "down":
                    self.thal_2 = 1.0
                else:
                    self.thal_2 = 0.0
            else:
                if checkbox.state == "down":
                    self.thal_3 = 1.0
                else:
                    self.thal_3 = 0.0

        # The options
        unknown_checkbox = MDCheckbox(group="Unknown", size_hint=(None, None), size=(48, 48))
        unknown_checkbox.bind(on_release=checkbox_callback)
        layout.add_widget(unknown_checkbox)

        unknown_label = Label(text="I don't know", size_hint=(None, None), size=(dp(300), dp(48)), valign='middle')
        unknown_label.color = option_color
        unknown_label.font_size = option_size
        layout.add_widget(unknown_label)

        fixed_defect_checkbox = MDCheckbox(group="Fixed Defect", size_hint=(None, None), size=(48, 48))
        fixed_defect_checkbox.bind(on_release=checkbox_callback)
        layout.add_widget(fixed_defect_checkbox)

        fixed_defect_label = Label(text="Fixed Defect", size_hint=(None, None), size=(dp(300), dp(48)), valign='middle')
        fixed_defect_label.color = option_color
        fixed_defect_label.font_size = option_size
        layout.add_widget(fixed_defect_label)

        normal_checkbox = MDCheckbox(group="Normal", size_hint=(None, None), size=(48, 48))
        normal_checkbox.bind(on_release=checkbox_callback)
        layout.add_widget(normal_checkbox)

        normal_label = Label(text="Normal", size_hint=(None, None), size=(dp(300), dp(48)), valign='middle')
        normal_label.color = option_color
        normal_label.font_size = option_size
        layout.add_widget(normal_label)

        reversible_defect_checkbox = MDCheckbox(group="Reversible Defect", size_hint=(None, None), size=(48, 48))
        reversible_defect_checkbox.bind(on_release=checkbox_callback)
        layout.add_widget(reversible_defect_checkbox)

        reversible_defect_label = Label(text="Reversible Defect", size_hint=(None, None), size=(dp(300), dp(48)), valign='middle')
        reversible_defect_label.color = option_color
        reversible_defect_label.font_size = option_size
        layout.add_widget(reversible_defect_label)


        #The submit button
        self.submit_button = Button(text='Next', on_press=self.next,size_hint = (0.9,0.1), pos_hint={"center_x": 0.5, "y": 0})
        layout.add_widget(self.submit_button)

        #Attach everything
        self.add_widget(layout)

    #Switch to the next screen
    def next(self,instance):

        global ans
        ans['thal_0.0'] = self.thal_0
        ans['thal_1.0'] = self.thal_1
        ans['thal_2.0'] = self.thal_2
        ans['thal_3.0'] = self.thal_3

        if('ans' in ans):
            del ans['ans']
        
        '''['age', 'thalach', 'oldpeak', 'ca', 'target', 'sex_0.0', 'sex_1.0',
       'cp_0.0', 'cp_1.0', 'cp_2.0', 'cp_3.0', 'exang_0.0', 'exang_1.0',
       'slope_0.0', 'slope_1.0', 'slope_2.0', 'thal_0.0', 'thal_1.0',
       'thal_2.0', 'thal_3.0']'''
        
        inp = [ans['age'],ans['thalach'],ans['oldpeak'],ans['ca'], ans['sex_0.0'], ans['sex_1.0'], ans['cp_0.0'], ans['cp_1.0'], ans['cp_2.0'], ans['cp_3.0'], ans['exang_1.0'], ans['slope_0.0'], ans['slope_1.0'], ans['slope_2.0'], ans['thal_0.0'], ans['thal_1.0'], ans['thal_2.0'], ans['thal_3.0']]

        model = torch.load('Heart_disease.CookieNeko').to(device)
        give = torch.tensor([inp],dtype=torch.float32)
        print(give)
        
        final = float(model(give))

        ans['ans'] = final


        print("Debug : Updated Struct -->",ans)
        self.res.update()
        self.manager.current = self.next_screen

class ResultScreen(Screen):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        global Q
        self.key = 'thal'
        self.next_screen = 'logo'
        #Layout
        layout = GridLayout(cols=1, spacing=30)
        layout.size_hint = (0.9,0.8)
        layout.pos_hint = {"center_x":0.5, "center_y":0.5}
        layout.padding = [30,0,30,50]

        #The writing, label
        self.text = Label(text="")
        self.text.font_size=35
        self.text.color = (1,0.4,0.4)
        layout.add_widget(self.text)

        #The submit button
        self.submit_button = Button(text='Try Again', on_press=self.next,size_hint = (0.3,0.1), pos_hint={"center_x": 0.1, "y": 0})
        self.submit_button.size = [0.1,0.1]
        layout.add_widget(self.submit_button)

        #Attach everything
        self.add_widget(layout)

    #Switch to the next screen
    def next(self,instance):

        self.manager.current = self.next_screen

    def update(self):
        per = ans['ans']*100
        self.text.text = "You have "+str(per)[:5]+"% probability of having a Heart Disease. \nTake good care."

class HeartDiseaseAppPredictor(App):

    def build(self):

        self.icon = 'Heartly_BareLOGO.ico'
        self.screen_manager = ScreenManager()
        self.theme_cls.primary_palette="Red"

        # create screens for each input field
        logo_screen = Logo(name='logo')
        age_screen = AgeScreen(name='age')
        gender_screen = GenderScreen(name='gender')
        cp_screen = CpScreen(name='cp')
        thallach_screen = ThalachScreen(name='thalach')
        exang_screen = ExangScreen(name ='exang')
        oldpeak_scren = OldpeakScreen(name ='oldpeak')
        slope_screen = SlopeScreen(name='slope')
        ca_screen = CaScreen(name='ca')
        result_screen = ResultScreen(name='result')
        thal_screen = ThalScreen(res = result_screen, name = 'thal')

        # add similar screens for each input field

        self.screen_manager.add_widget(logo_screen)
        self.screen_manager.add_widget(age_screen)
        self.screen_manager.add_widget(gender_screen)
        self.screen_manager.add_widget(cp_screen)
        self.screen_manager.add_widget(thallach_screen)
        self.screen_manager.add_widget(exang_screen)
        self.screen_manager.add_widget(oldpeak_scren)
        self.screen_manager.add_widget(slope_screen)
        self.screen_manager.add_widget(ca_screen)
        self.screen_manager.add_widget(thal_screen)
        self.screen_manager.add_widget(result_screen)

        # add similar screens for each input field

        return self.screen_manager

if __name__ == "__main__":
    HeartDiseaseAppPredictor().run()