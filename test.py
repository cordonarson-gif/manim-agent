from manim import *

class TestScene(Scene):
    def construct(self):
        # 1. 创建一个蓝色的圆
        circle = Circle()
        circle.set_fill(BLUE, opacity=0.5)
        
        # 2. 创建一个数学公式（测试 LaTeX 环境）
        text = MathTex("E = mc^2")
        text.next_to(circle, RIGHT)
        
        # 3. 播放动画
        self.play(Create(circle))
        self.play(Write(text))
        self.wait(1)
