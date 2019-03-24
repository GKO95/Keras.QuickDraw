from keras.models import load_model

# Config 모듈을 적용하기 위해서는 다른 Kivy 모듈을 불러오기 전에 설정을 해야 한다.
# 어플리케이션 창을 높이와 너비가 800px로 고정된 사이즈 재조정 불가능한 테두리 없는 창으로 설정한다.
from kivy.config import Config
Config.set('graphics', 'resizable', '0'); Config.set('graphics', 'borderless', '1')
Config.set('graphics', 'height', '600'); Config.set('graphics', 'width', '600')

from kivy.clock import Clock
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen, SlideTransition
from kivy.uix.widget import Widget
from kivy.graphics import Color, Line, Ellipse
from kivy.properties import StringProperty, NumericProperty

import numpy as np
import cv2
import os
import time
import sys

# 학습된 모델 활용을 위해 해당 스크립트에 로드합니다.
model = load_model('.\\model_dir\\QD_model.h5')

# 클래스를 자동으로 추출하도록 시행되는 코드입니다. 우선 클래스의 이름을 어디에서 참고할 것인지 경로를 지정합니다.
path = os.listdir(".\\numpy_dataset")
class_name = {}
class_count = 0
# For 문을 통해 해당 경로에 있는 모든 파일의 이름에 대하여 전부 확인합니다.
for file in path:
    # 우선 데이터세트 파일 이름을 통해 경로를 포함한 파일 이름 전체를 획득합니다.
    file = ".\\numpy_dataset\\" + file
    # splitext 매소드를 통해 확장자 텍스트를 제거합니다.
    file_name, _ = os.path.splitext(file)
    # 공통적으로 들어있는 앞 부분을 전부 제거하여 클래스의 이름만 추출하여 사전에 추가합니다.
    class_name[class_count] = file_name[38:]
    # 사전에 추가를 완료하였으면 다음 클래스 이름을 받기 위한 키를 할당합니다.
    class_count += 1

# 클래스 인덱스 순서를 무작위로 섞으되 반복되지 않도록 range를 설정하고 리스트에 재할당한 것입니다.
rand_int = np.arange(class_count-1)
np.random.shuffle(rand_int)

Builder.load_string("""
<Screen>:
    canvas:
        Color:
            rgb: 255, 255, 255
        Rectangle:
            size:self.size

<SettingsScreen>:
    id : div_screen
    ProjectCanvas:
        id: div_canvas
        canvas.before:
            Color:
                rgb: 255, 255, 255
            Rectangle:
                size: self.size
                pos: self.pos
    FloatLayout:
        Button:
            text: 'REDO'
            color: 0, 0, 0, 0.5
            size_hint: (.1, .1)
            pos_hint: {'right':0.3}
            background_color: (0, 0, 0, 0.1)
            on_release: div_canvas.redo()
        Button:
            text: 'UNDO'
            color: 0, 0, 0, 0.5
            size_hint: (.1, .1)
            pos_hint: {'right':0.2}
            background_color: (0, 0, 0, 0.1)
            on_release: div_canvas.undo()
        Button:
            text: 'CLEAR'
            bold: 1
            color: 1, 0, 0, 1
            size_hint: (.1, .1)
            pos_hint: {'right':0.1}
            background_color: (0, 0, 0, 0.1)
            on_release: div_canvas.new_canvas()
        Label:
            text: div_canvas.pred_realtime
            bold: 1
            font_size: 26
            color: 1, 0, 0, 1
            size_hint: (.1, .1)
            pos_hint: {'right':.95,'top':1}
            background_color: (0, 0, 0, 0)
        Label:
            text: str(app.time)
            color: 1, 0, 0, 1
            size_hint: (.1, .1)
            pos_hint: {'right':1}
            background_color: (0, 0, 0, 0)

<ProjectReady>:
    Button:
        text: root.show_keyword
        # 버튼에 마우스 클릭을 떼면 App 클래스, 즉 ProjectMain 내의 load_canvas() 클래스 매소드를 시행합니다.
        on_release: app.load_canvas()

<ProjectEnd>:
    BoxLayout:
        canvas.before:
            Color:
                rgba: 0, 0, 0, 0.62
            Rectangle:
                pos: self.pos
                size: self.size
        orientation: 'vertical' 
        Label: 
            text: 'CORRECT: ' + str(app.correct)
        BoxLayout:
            orientation: 'horizontal'
            size_hint_y:0.5
            Button:
                text: 'FINISH'
                on_release: app.end()
            Button:
                text: 'AGAIN'
                on_release: app.again()

<ProjectCorrect>:
    Button:
        text:'CORRECT'
        on_release: app.load_ready()

<ProjectWrong>:
    Button:
        text:'WRONG'
        on_release: app.load_ready()
""")


class SettingsScreen(Screen):
    pass


# ProjectCanvas() 클래스는 Screen()를 슈퍼클래스로 두는 서브클래스입니다.
# 프로그램에서 실제로 그림을 그릴 수 있는 틀인 "캔버스"를 제공합니다.
# 본 클래스는 ProjectMain()이라는 클래스의 self.canvas 속성에 할당되었습니다.
class ProjectCanvas(Widget):

    # drawing 변수는 그림의 한 점 또는 한 획에 대한 (x,y) 좌표 정보의 1차원 행렬을 담는 2차원 NumPy 행렬이다.
    drawing = []

    # undolist 변수는 undo 버튼을 클릭하였을 때, 되돌리기 기능의 redo 버튼을 대비한 임시 정보 저장을 위한 2차원 NumPy 행렬입니다.
    # undolist 변수가 담는 정보는 한 획의 (x,y) 좌표 정보가 있는 1차원 행렬입니다.
    undolist = []

    # 실시간으로 추론 결과를 화면에 보여지도록 하는 String 전용 변수입니다.
    pred_realtime = StringProperty()

    # ProjectCanvas()의 __init__은 속성 초기값을 설정하며, 직접 self.attribute_name 등을 통해 속성을 만들 수도 있습니다.
    # 어떠한 값이 입력될 것인지 알지만 아직 정의되지 않은 파라미터는 "**kwargs" (i.e. known arguments)라고 합니다.
    # 그리고 여러 입력을 받을 수 있도록 하는 파라미터는 "*args"라고 합니다.
    def __init__(self, **kwargs):
        # ProjectCanvas()가 속해있는 Screen() 슈퍼클래스의 __init__ 속성을 모두 ProjectCanvas().__init__에 불러옵니다.
        # PYTHON 2에서는 아래의 코드를 사용하나, PYTHON 3에서는 코드를 super().__init__(**kwargs)로 간략화시킬 수 있습니다.
        super(ProjectCanvas, self).__init__(**kwargs)
        self.pred_realtime = ''

    # 마우스로 캔버스를 클릭할 때 캔버스에 점을 찍을 수 있도록 지정하며, 직선의 시작점을 제공하기도 합니다.
    def on_touch_down(self, touch):
        with self.canvas:
            # 캔버스에 그릴 색과 획의 두께를 결정합니다.
            Color(0, 0, 0); d = 16
            # 마우스 클릭 한 번으로 점을 찍으며 <Ellipse object>. 이를 drawing 변수에 추가합니다.
            self.drawing.append(Ellipse(pos=(touch.x - d / 2, touch.y - d / 2), size=(d, d)))
            # 마우스 클릭을 하고 움직일 때 <Line object> 좌표 정보를 drawing 변수에 추가합니다.
            # 즉, 한 번의 마우스 클릭은 <Ellipse object>와 <Line object>의 좌표 정보를 동시에 저장합니다.
            self.ud = Line(points=(touch.x, touch.y), width=8)
            self.drawing.append(self.ud)

    # 마우스가 클릭된 상태에서 캔버스 위에 움직였을 때, 직선을 그릴 수 있도록 합니다.
    def on_touch_move(self, touch):
        """전체적인 이동경로 좌표를 보고 싶으면 해당 함수의 끝부분에 print(touch.ud['line'])를 추가하도록 합니다.
        만일 시좌표(instantaneous coordinate)을 보고 싶으면 대신 print(np.array(Window.mouse_pos))을 추가합니다."""
        # 아래의 코드가 활성화되지 않으면 선을 그릴 수가 없습니다.
        # 여기에 저장된 데이터로 시작점과 불연속점, 그리고 끝점을 모두 입력받아 실시간으로 선을 캔버스에 나타내는 방식입니다.
        self.ud.points += [touch.x, touch.y]

        # 실시간 데이터 처리가 가장 이상적이지만, 현재 추출하고자 하는 데이터는 막대한 픽셀양의 PNG 이미지이다.
        # 실시간 이미지 처리에 최적화되지 않은 KIVY에서는 실시간 데이터 처리는 오히려 심각한 문제를 야기한다.
        # 어차피 그림의 시작과 끝에 PNG 추출이 이루어지므로, 획이 특정 데이터양이 모일 때만 PNG를 추출한다.
        if len(self.ud.points) % 100 is 0:
            self.capture_canvas()

    # 마우스 클릭을 뗀 상태에서 직선 정보보다 PNG 이미지 추출이 중점으로 여긴다.
    def on_touch_up(self, touch):
        # 좌표 평면을 행렬화하여 계산하는 것보다 직접 캔버스에 그려진 이미지를 추출하여 활용한다.
        # 수학적으로 좌표평면 위치들을 행렬화시켜 활요하는 것은 픽셀레이션 및 각 픽셀에 대한 회색조 값에 대하여
        # 표현하는데 더 복잡해지는 문제가 생긴다.
        self.capture_canvas()

    # Undo 버튼을 클릭할 시 실행되는 함수입니다.
    def undo(self):
        # 만일 그림을 그린게 없으면 undo 함수는 아무런 기능을 하지 않습니다.
        if len(self.drawing) == 0:
            return
        # <Ellipse object> 및 <Line object> 좌표 정보를 임시 변수에 추출합니다.
        tmp_ellipse = self.drawing.pop(-2)
        tmp_line = self.drawing.pop(-1)
        # Undo 및 Redo 기능을 위해 좌표 정보를 undolist 리스트 변수에 저장합니다.
        self.undolist.append(tmp_ellipse)
        self.undolist.append(tmp_line)
        # 캔버스에 저장된 해당 좌표 정보들을 삭제합니다.
        self.canvas.remove(tmp_ellipse)
        self.canvas.remove(tmp_line)
        # 삭제하여 변경된 그림을 추론을 위해 저장합니다.
        self.capture_canvas()

    # Redo 버튼을 클릭할 시 실행되는 함수입니다.
    def redo(self):
        # 만일 그림을 그린게 없으면 redo 함수 또한 아무런 기능을 하지 않습니다.
        if len(self.undolist) == 0:
            return
        # undolist 리스트 변수에 임시로 저장된 <Ellipse object> 및 <Line object> 좌표 정보를 임시 변수에 추출합니다.
        tmp_ellipse = self.undolist.pop(-2)
        tmp_line = self.undolist.pop(-1)
        # Undo 및 Redo 기능을 위해 좌표 정보를 drawing 리스트 변수에 저장합니다.
        self.drawing.append(tmp_ellipse)
        self.drawing.append(tmp_line)
        # 캔버스에 저장된 해당 좌표 정보들을 추가합니다.
        self.canvas.add(tmp_ellipse)
        self.canvas.add(tmp_line)
        # 추가하여 변경된 그림을 추론을 위해 저장합니다.
        self.capture_canvas()

    # Clear 버튼을 눌렀을 때 캔버스와 모든 리스트 변수도 모두 초기화시키고 화면에 보여질 실시간 추론 결과도 공백으로 바꿉니다.
    def new_canvas(self):
        self.canvas.clear()
        self.drawing.clear()
        self.undolist.clear()
        self.pred_realtime = ''

    # PNG 이미지 저장을 할 떄마다 예측모델이 이미지 추론을 진행합니다.
    def capture_canvas(self):
        # 이미지를 CANVAS.PNG 이름으로 저장합니다.
        self.export_to_png(".\\image\\CANVAS.PNG")
        # png2numpy() 함수를 통해 변환된 NumPy 행렬을 입력 데이터로 넣기 위해 4차원으로 재조정합니다.
        processed = png2numpy().reshape(1, 28, 28, 1)
        # 입력 데이터에 대한 각 클래스의 추론 확률을 불러옵니다.
        self.pred_probab = model.predict(processed)[0]
        # 가장 높은 확률의 추론 데이터의 클래스를 확인합니다.
        self.pred_class = list(self.pred_probab).index(max(self.pred_probab))
        print("실시간 추론 결과: [1위] " + class_name[self.pred_class] + " - " + str(max(self.pred_probab)))
        self.pred_realtime = class_name[self.pred_class]


# PNG 확장자를 모델 입력 데이터로 최적화시킨 NumPy로 변환시키는 함수입니다.
def png2numpy():
    # PNG 이미지 파일을 NumPy 행렬로 변환시키며, 각 픽셀은 RGB 값으로 할당된다.
    img = cv2.imread('.\\image\\CANVAS.PNG')
    # RGB 값들로 구성된 NumPy 행렬을 회색조 값으로 구성되 NumPy 행렬로 변환합니다.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 하얀색 배경이 255 값으로 할당되어 있어, 아래의 코드는 값들을 반전시키며 np.uint8을 통해 최대값을 1로 변환시킵니다.
    img = (img < 128).astype(np.uint8)

    # coords는 행과 열이 영벡터가 아닌 동안의 벡터의 시작점을 [x, y] 행렬로 값을 추출하고, 이들의 값 차이가 바로 너비와 높이가 된다.
    coords = cv2.findNonZero(img)
    # 그림이 시작되는 x값, y값, 그림의 너비 w 및 높이 h를 획득합니다.
    x, y, w, h = cv2.boundingRect(coords)

    # 너비와 높이의 크기를 비교하여 그림을 중심에 위치할 수 있도록 설정합니다.
    if h > w:
        norm = np.zeros((h + 2, h + 2), dtype=np.float32)
        norm[1:-1, int((h - w) / 2) + 1:int((h - w) / 2) + 1 + w] = img[y:y + h, x:x + w]
    else:
        norm = np.zeros((w + 2, w + 2), dtype=np.float32)
        norm[int((w - h) / 2) + 1:int((w - h) / 2) + 1 + h, 1:-1] = img[y:y + h, x:x + w]

    # 동일한 비율로 cropping 된 그림을 모델 입력 데이터에 맞게 1px 테두리 패딩의 28x28 픽셀 대응 2차원 NumPy 행렬로 변환시킨다.
    img = cv2.resize(norm, (26, 26), interpolation=cv2.INTER_AREA)
    img = np.pad(img, pad_width=1, mode='constant', constant_values=0)
    return img


# 캔버스에 그림을 그리기 전에 주제어를 제공하는 등의 스크린을 담당합니다.
class ProjectReady(Screen):
    # KIVY 언어 내에서 할당된 변수를 아래와 같은 방식으로 할당한다; string 형식.
    show_keyword = StringProperty()

    def __init__(self, **kwargs):
        super(ProjectReady, self).__init__(**kwargs)
        # 무작위로 섞인 키워드를 순서대로 반복없이 선별하는 count 속성을 통해 초기 주제어를 설정합니다.
        self.count = 0
        self.show_keyword = class_name[rand_int[self.count]]

    def update_keyword(self):
        # 다음 순서의 키워드를 선별하기 위해 count 속성에 +1을 하고, 새로운 키워드를 가져옵니다.
        self.count += 1
        self.show_keyword = class_name[rand_int[self.count]]


# 게임이 종료될 때 나타나는 스크린 화면에 대한 클래스입니다.
class ProjectEnd(Screen):
    pass


# 캔버스에 그림을 그려 컴퓨터가 맞은 추론을 하였을 때 나타나는 스크린 화면에 대한 클래스입니다.
class ProjectCorrect(Screen):
    pass


# 캔버스에 그림을 그려 컴퓨터가 제 시간 내에 추론을 하지 못하였을 때 나타나는 스크린 화면에 대한 클래스입니다.
class ProjectWrong(Screen):
    pass


# KIVY 형식의 어플리케이션을 실제로 실행가능하도록 수행하는 App 클래스로 형성된 ProjectMain() 클래스입니다.
# 어플리케이션의 주된 기능 및 함수들은 이곳에 포함되어 개발됩니다.
class ProjectMain(App):
    # 실시간으로 추론 결과를 화면에 보여지도록 하는 String 전용 변수입니다.
    pred_realtime = StringProperty()

    # 맞춘 문제 개수를 카운트 하기 위한 숫자 전용 변수입니다.
    correct = NumericProperty()

    # 카운트다운을 위한 숫자 전용 변수입니다.
    time = NumericProperty()

    rand_int_count = NumericProperty()

    # ProjectMain()의 __init__은 속성 초기값을 설정하며, 직접 self.attribute_name 등을 통해 속성을 만들 수도 있습니다.
    def __init__(self, **kwargs):
        # ProjectMain()이 속해있는 App() 슈퍼클래스의 __init__ 속성을 모두 ProjectMain().__init__에 불러옵니다.
        super(ProjectMain, self).__init__(**kwargs)
        # count 변수는 문제의 개수를 카운트하기 위한 것입니다.
        self.count = 0
        self.correct = 0
        self.pred_realtime = ''
        self.start = 0
        self.time = 0

    # build() 매소드는 App 클래스 내에 작성 및 내포된 코드를 통해 어플리케이션을 실행시킵니다.
    # return 값을 특정 위젯으로 지정하면, 해당 위젯을 root 위젯으로 설정하고 KIVY 창에 나타납니다.
    def build(self):
        # KIVY 어플리케이션 창 이름을 지정합니다.
        self.title = 'KIVY Canvas'
        # ProjectReady() 및 ProjectCanvas() 클래스를 ProjectMain() 클래스 내의 속성(attribute)에 정의합니다.
        # ProjectReady() 및 ProjectEnd()은 Screen 클래스이며, 각자 이름은 'ready'와 'end'입니다.
        # ProjectCorrect() 및 ProjectWrong()은 Screen 클래스이며, 각자 이름은 'got_correct'와 'got_wrong'입니다.
        # 이는 ScreenManager() 클래스가 스크린을 스크린 속성의 name 을 통해 제어/관리를 하기 때문입니다.
        self.ready = ProjectReady(name='ready')
        self.end = ProjectEnd(name='end')
        self.got_correct = ProjectCorrect(name='got_correct')
        self.got_wrong = ProjectWrong(name='got_wrong')
        self.canvas = SettingsScreen(name='canvas')

        # ProjectReady()의 카운팅 속성에 접속을 하여 수정을 가능토록 하며, 게임을 새로 시작할 시 필요합니다.
        self.ready.count = 0

        # ProjectMain() 클래스에 0.35초만에 화면을 슬라이드 천이이동시키는 transition 속성을 부여합니다.
        self.transition = SlideTransition(duration=0.35)
        # 이름 "root"라는 ScreenManager()은 어플리케이션 내에서 여러 스크린을 제어 및 관리하기 위해 사용됩니다.
        # 그 중에서 "root"의 스크린 천이방식은 transition 속성을 통해 이루어집니다(즉, 0.35초만의 슬라이드 천이).
        root = ScreenManager(transition=self.transition)
        # ProjectReady() 및 End(), Correct()과 Wrong()을 속성을 통해" root"에 넣어 ScreenManager()로 제어가 가능합니다.
        root.add_widget(self.ready)
        root.add_widget(self.end)
        root.add_widget(self.got_correct)
        root.add_widget(self.got_wrong)
        root.add_widget(self.canvas)

        # 클래스 매소드 self.Is_predicting을 매초마다 실행하며, 본 클래스 매소드가 추론이 옳고 그름을 결정한다.
        Clock.schedule_interval(self.is_predicting, 1)

        # build() 메소드의 return 값이 "root"로 정해지며, 이제 해당 어플리케이션의 뿌리(root)는 "root"입니다.
        return root

    # ProjectCanvas() 화면으로 전환하였을 경우, 20초
    def is_predicting(self, *args):
        # 타이머를 실행시키는데, 계산법은 CPU 에서 지속적으로 측정된 클래스 시행 시간을 self.start 라는 캔버스를 불러왔는데 걸린
        # 동일한 CPU 측정 시간을 빼서 정수화 시킨 것이다. Clock.schedule_interval(self.is_predicting, 1) 문으로 인해 매초마다
        # int(time.time() - self.start)은 1초씩 증가하게 되는데, 이를 20초에 빼서 20초 타이머로 만들었다.
        self.time = 20 - int(time.time() - self.start)

        # 본격적인 추론 작업은 오로지 ProjectCanvas() 스크린이 현재 화면에 나탄났을 때에만 진행한다.
        if self.root.current == 'canvas':
            # 타이머가 0이 되기 전에 경우에서...
            if self.time != 0:
                # 추론 결과가 키워드와 동일하면...
                if self.canvas.children[1].pred_realtime == self.ready.show_keyword:
                    # 맞춘 문제 카운트를 +1 하고, 정답 맞춤 화면인 ProjectCorrect() 스크린으로 전환합니다.
                    self.correct += 1
                    self.rand_int_count += 1
                    self.transition.direction = 'right'
                    self.root.current = 'got_correct'
            # 그러나 타이머가 0이 되었을 시...
            else:
                # 때마침 추론 결과가 키워드와 동일할 경우를 대비하여...
                if self.canvas.children[1].pred_realtime == self.ready.show_keyword:
                    # 맞춘 문제 카운트를 +1 하고, 정답 맞춤 화면인 ProjectCorrect() 스크린으로 전환합니다.
                    self.correct += 1
                    self.rand_int_count += 1
                    self.transition.direction = 'right'
                    self.root.current = 'got_correct'
                # 하지만 추론 결과가 키워드와 전혀 동일하지 않으면...
                else:
                    # 정답 틀림 화면인 ProjectWrong() 스크린으로 전환합니다.
                    self.rand_int_count += 1
                    self.transition.direction = 'right'
                    self.root.current = 'got_wrong'

    # 대기화면 ProjectReady()에서 'start' 버튼을 누르면 대기화면을 치우도록 시행되는 클래스 매소드입니다.
    def load_canvas(self):
        # 스크린 천이이동을 왼쪽으로 가게 합니다.
        self.transition.direction = 'left'

        # count 속성에 +1을 하여 카운트합니다.
        self.count += 1
        # start 속성은 20초 시간 카운트다운을 위한 것입니다.
        self.start = time.time()

        # ScreenManager()의 current 속성은 "root"에 현재 보여질 화면을 의미합니다.
        # 그러므로 "root"의 현재 화면을 ProjectCanvas()로 설정하여, 버튼을 클릭하면 바로 캔버스가 나타나게 합니다.
        self.root.current = 'canvas'

    # 캔버스 화면 ProjectCanvas()에서 대기화면 ProjectReady()로 화면 전환을 하는데 시행되는 클래스 매소드입니다.
    def load_ready(self):
        # 만일 count 속성으로 카운팅된 숫자가 5, 즉 5개의 문제를 모두 풀었으면 load_end() 클래스 매소드를 시행합니다.
        if self.count == 5:
            self.load_end()
        # 만일 5개의 문제를 풀지 않은 상태일 때 시행됩니다.
        else:
            # 대기화면에 나타날 키워드를 update_keyword() 클래스 매소드를 통해 임의로 새로 할당합니다.
            self.ready.update_keyword()
            # 기준의 캔버스를 새로운 캔버스로 덮어씁니다.
            self.canvas.children[1].new_canvas()
            # 화면 전환을 오른쪽 방향으로 진행합니다.
            self.transition.direction = 'right'
            # 현재 보여질 화면의 이름은 'ready'입니다.
            self.root.current = 'ready'

    # ProjectEnd()를 불러오는 클래스 매소드입니다.
    def load_end(self):
        self.transition.direction = 'right'
        self.root.current = 'end'

    # ProjectEnd() 스크린에서 AGAIN 버튼을 눌렀을 때, 문제를 처음부터 다시 풀도록 모든 속성을 초기화시키는 클래스 매소드입니다.
    def again(self):
        # 모든 것(캔버스, 카운팅, 키워드 등)들을 초기화시킵니다.
        self.canvas.children[1].new_canvas()
        self.count = 0
        self.correct = 0
        self.ready.update_keyword()
        self.transition.direction = 'right'
        self.root.current = 'ready'
        self.ready.count = 0

        # 무작위로 섞은 인덱스를 다시 한 번 무작위로 섞습니다.
        np.random.shuffle(rand_int)

    # ProjectEnd() 스크린 화면에서 FINISH 버튼을 눌렀을 때, 시스템을 종료하는 정적 클래스 매소드(self 구성이 없는 함수)입니다.
    @staticmethod
    def end():
        sys.exit(1)


# 본 스크립트가 외부 스크립트가 아닌 주요 스크립트로 실행될 때에만 위의 프로그램을 정상적으로 실행시킵니다.
if __name__ == '__main__':
    # run() 매소드는 개발환경 필요없이 독립적으로 실행할 수 있도록 합니다.
    ProjectMain().run()
