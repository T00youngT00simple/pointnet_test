print('Here\n')

print(123 + 222)

D = {'food': 'Spam', 'quantity': 4, 'color': 'pink'}
print(D['food'])

D['quantity'] += 1

print(D)

squares = [x ** 2 for x in [1, 2, 3, 4, 5]]
print(squares)

f = open('data.txt', 'w')
f.write('Hello\n') # Make a new file in output mode ('w' is write)
# Write strings of characters to it
f.write('world\n') # Return number of items written in Python 3.X
f.close()


f = open('data.txt')
text = f.read()
print(text)
A=text.split()
print(A)

for line in open('data.txt'): print(line)

X = set('spam')
Y ={'h','a','m'}
print(X,Y)

{n**2 for n in [1,2,3,4]}

print(list(set([1,2,1,3,1])))
print(set('spam')-set('ham'))