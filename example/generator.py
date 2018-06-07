def fibonacci_func(n) :
	a,b = 0,1
	i = 0
	while True :
		if(i > n) : return
		yield a
		a,b = b, a+b
		i += 1

fib = fibonacci_func(10)
print fib #generator

for x in fib :
	print x
