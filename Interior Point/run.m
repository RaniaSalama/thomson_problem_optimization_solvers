n=20
k=2
x0=rand(n*k,1)'
x = fmincon(@func,x0,[],[],[],[],[],[],@constraints)
xshapped = reshape(x,[k,n])
plot(xshapped(1,:), xshapped(2,:), 'o')