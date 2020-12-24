
class Test:
    def __init__(self):
        self.v = 1.0

    @staticmethod
    def get_v():
        return 2.0
        
    def get_v(self):
        return self.v



t = Test()

print(t.get_v())
print(Test.get_v())