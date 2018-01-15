@profile
def fn():
  l = []
  for _ in range(1000000):
    l.append('hoge')
    l.pop()
fn()
