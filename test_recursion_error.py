class A:
    def __getattribute__(self, name):
        return getattr(self, name)


a = A()
print("Testing Recursion Limit")
try:
    a.test
except RecursionError:
    print("Recursion Limit ok")
