# Syntax of the command:
#------------------------------------
#  python movielens.py fill_by_type 
#------------------------------------
#
#  'fill_by_type' can be 'fill_by_movie' or 'fill_by_user'
#  allows you to custom the method for filling the R matrix
#  when a film was not rated by an user.




from csv import *
from numpy import *
from sys import argv
import matplotlib.pyplot as plt
import pylab as pl


#====================== check of the arguments =========================
if argv[1]!='fill_by_user' and argv[1]!='fill_by_movie':
  print 'Invalid filling method. Please use one of the following arguments:\nfill_by_movie\nfill_by_user'
  print '\nsyntax: python movielens.py fill_method'
  exit()


#====================== load the data ==================================
print 'loading data...'
with open('u1.base', 'rb') as f:
  fieldnames=['user','movie','rating','datestamp']
  reader = DictReader(f,delimiter='\t', fieldnames=fieldnames)
  baseU1 = [dict([key, int(value)] for key, value in row.iteritems()) for row in list(reader)]

with open('u1.test', 'rb') as f:
  fieldnames=['user','movie','rating','datestamp']
  reader= DictReader(f,delimiter='\t', fieldnames=fieldnames)
  testU1 =  [dict([key, int(value)] for key, value in row.iteritems()) for row in list(reader)]

with open('u.user', 'rb') as f:
  reader = DictReader(f, delimiter = '|', fieldnames=['user','age','sex','occupation','zipcode'])
  Users = list(reader)

with open('u.genre', 'rb') as f:
  reader = DictReader(f, delimiter = '|', fieldnames=['genre','genrecode'])
  Genres = list(reader)

with open('u.item', 'rb') as f:
  reader = DictReader(f, delimiter = '|', fieldnames=['itemcode','name','releasedate','imdb','unknown','Action','Adventure','Animation','Children','Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western'])
  Items = list(reader)

with open('u.occupation', 'rb') as f:
  reader = DictReader(f, fieldnames=['occupation'])
  Occupations = list(reader)
Nmovies=len(Items)
Nusers=len(Users)


#====================== Build a user-item matrix R =====================
print 'building R matrix...'
R=zeros((Nusers,Nmovies))
b=0
for rating in baseU1:
  R[rating['user']-1,rating['movie']-1]=rating['rating']
print "R built.\n"


#====================== Compute the means ==============================
print "computing rating means on each movie..."
movieMeans=[]
for movie in R.T:
  if sum(movie)!=0:
    movieMeans.append(mean([ movie[i] for i in xrange(len(movie)) if movie[i]!=0 ]))
  else:
    movieMeans.append(0)  #If the movie has not been rated by anyone.

print "computing rating means on each user..."
userMeans=[]
for user in R:
  userMeans.append(mean([ user[i] for i in xrange(len(user)) if user[i]!=0 ]))





#==================== Fill both matrices Rc and Rr =====================
print 'filling R (both versions)...'
Rr=array([ [R[i,j] for j in xrange(Nmovies)] for i in xrange(Nusers) ])
Rc=array([ [R[i,j] for j in xrange(Nmovies)] for i in xrange(Nusers) ])

for i,movie in enumerate(R.T):
  for j, rating in enumerate(movie):
    if rating==0:
      Rc[j,i]=movieMeans[i]

for i,user in enumerate(R):
  for j, rating in enumerate(user):
    if rating==0:
      Rr[i,j]=userMeans[i]

print 'Rc (by movie) and Rr (by user) filled.\n'


"""

#====================== Plot the matrices ==============================

plt.imshow(R, interpolation='none')
plt.ylabel('User Id')
plt.xlabel('Film Id')
plt.title('R not filled')
plt.show()

plt.imshow(Rc, interpolation='none')
plt.ylabel('User Id')
plt.xlabel('Film Id')
plt.title('Filled with the movie\'s ratings mean')
plt.show()

plt.imshow(Rr, interpolation='none')
plt.ylabel('User Id')
plt.xlabel('Film Id')
plt.title('Filled with the user\'s ratings mean')
plt.show()



#====================== compute the SVD ================================
print 'computing SVD decomposition with %s method:'%argv[1]
if argv[1]=='fill_by_movie':
  u,diags,v= linalg.svd(Rc, full_matrices=0)
  s=diag(diags)
  print 'SVD: allclose(Rc,U.S.V\'):',allclose(Rc,dot(u,dot(s,v)))
if argv[1]=='fill_by_user':
  u,diags,v= linalg.svd(Rr, full_matrices=0)
  s=diag(diags)
  print 'SVD: allclose(Rr,U.S.V\'):',allclose(Rr,dot(u,dot(s,v)))



remindMAE=[]
for k in xrange(1,31):
  #Approximation of R by keeping only k singular values:
  #keep only the k first columns of u and the k first lines of v
  Sk=s[:k,:k]
  Uk=transpose(u.T[:k])
  Vk=v[:k]
  print 'approximation computed with only %d singular values.'%k
  plt.title('Approximation of R with only %d singular values'%k)
  plt.xlabel('Film Id')
  plt.ylabel('User Id')
  #plt.imshow(dot(Uk,dot(Sk,Vk)),interpolation='none')
  #plt.show()



  #Compare the predictions with the real ratings:
  MAE=0
  for dic in testU1:
    prediction=dot( Uk[dic['user']-1,:] , dot( Sk , Vk[:,dic['movie']-1] ))
    MAE += abs( prediction - dic['rating'] )
  MAE /= float(len(testU1))
  print "MAE= %f\n"%MAE
  remindMAE.append(MAE)



  #Build the feature-matrices Xk and Yk:
  sqrtSk=diag(map(sqrt,diags[:k]))
  Xk = dot(Uk , sqrtSk )
  Yk = dot(sqrtSk , Vk)




#====================== Find the best approximation ====================
k=range(1,31)
plt.plot(k,remindMAE)
plt.plot([0],[0.79])
plt.plot([0],[0.835])
plt.ylim(0.79,0.84)
plt.title('Mean Absolute Error=f(k)')
plt.xlabel('Number of singular values kept in the approximation of R')
plt.ylabel('MAE')
plt.show()


#====================== Compare the predictions ========================
Sk=s[:12,:12]
Uk=transpose(u.T[:12])  #Let's use k=12 for this part
Vk=v[:12]
print '\nComparing predictions between approximation and naive method...'
predictions_svd=[]
predictions_naive=[]
for i in xrange(Nmovies):
  for u in xrange(Nusers):
    predictions_svd.append(   dot( Uk[u-1,:] , dot( Sk , Vk[:,i-1] ))   )
    if argv[1]=='fill_by_user':
      predictions_naive.append(  Rr[u-1,i-1]  )
    if argv[1]=='fill_by_movie':
      predictions_naive.append(  Rc[u-1,i-1]  )
def absdiff(a,b):
  return abs(a-b)
differences=map(absdiff , predictions_svd , predictions_naive)
print 'The mean difference between the SVD approximation and the true naive method is of %f stars.'%(mean(differences))

"""


#====================== Ordinary Least Squares =========================
W=R>0

k=3
lambda_=0.02
nbrIter=10
X=ones((Nusers,k), dtype=float)
Y=ones((k,Nmovies), dtype=float)

remindMAE=[]

for iteration in range(nbrIter):

  for u in xrange(Nusers):
    YWu = dot(Y , diag(W[u]))
    A = dot( YWu, Y.T) + lambda_*eye(k)
    b = dot( YWu, (R[u]).T )
    X[u,:] = linalg.solve( A , b ).T
  for i in xrange(Nmovies):
    XTWi = dot( X.T , diag(W[:,i]) )
    A = dot( XTWi , X ) + lambda_*eye(k)
    b = dot( XTWi , R[:,i] )
    Y[:,i] = linalg.solve(  A , b )

  MAE=0
  for dic in testU1:
    prediction=dot( X[dic['user']-1,:] , Y[:,dic['movie']-1] )
    MAE += abs( prediction - dic['rating'] )
  MAE /= float(len(testU1))
  print "MAE= %f\n"%MAE
  remindMAE.append(MAE)


pl.plot(range(nbrIter),remindMAE)
pl.title('Mean Absolute Error = f(nbIter)')
pl.suptitle('MAE with predictions based on the product X&Y')
pl.xlabel('nbrIter')
pl.ylabel('MAE')
print min(remindMAE)
pl.show()


