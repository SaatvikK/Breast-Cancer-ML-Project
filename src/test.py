class A:
  @staticmethod
  def funcstat():
    print("static")
  
  def funcclass(self):
    A.funcstat()

myA = A()
myA.funcclass()