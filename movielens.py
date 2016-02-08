# Syntax of the command:
#------------------------------------
#  python movielens.py fill_by_type 
#------------------------------------
#
#  'fill_by_type' can be 'fill_by_movie' or 'fill_by_user'
#  allows you to custom the method for filling the R matrix
#  when a film was not rated by an user.



from csv import *
from time import time
from numpy import *
from sys import argv
import matplotlib.pyplot as plt
import pylab as pl



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



#~ '''
#====================== check of the arguments =========================
if len(argv)==1:
  print 'Please chose a filling method as argument:\nfill_by_movie\nfill_by_user'
  print '\nsyntax: python movielens.py fill_method'
  exit('No filling method')
elif argv[1]!='fill_by_user' and argv[1]!='fill_by_movie':
  print 'Invalid filling method. Please use one of the following arguments:\nfill_by_movie\nfill_by_user'
  print '\nsyntax: python movielens.py fill_method'
  exit('Wrong filling method')



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
#~ '''


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
"""

#~ '''
#====================== compute the SVD ================================
start_time = time()
print 'computing SVD decomposition with %s method:'%argv[1]
if argv[1]=='fill_by_movie':
  u,diags,v= linalg.svd(Rc, full_matrices=0)
  s=diag(diags)
  print 'SVD: allclose(Rc,U.S.V\'):',allclose(Rc,dot(u,dot(s,v)))
if argv[1]=='fill_by_user':
  u,diags,v= linalg.svd(Rr, full_matrices=0)
  s=diag(diags)
  print 'SVD: allclose(Rr,U.S.V\'):',allclose(Rr,dot(u,dot(s,v)))


#========================= Compute X and Y =============================
k = 12
Sk=s[:k,:k]
Uk=transpose(u.T[:k])
Vk=v[:k]
print 'approximation computed with only %d singular values.'%k
sqrtSk=diag(map(sqrt,diags[:k]))
X= dot(Uk , sqrtSk )
Y = dot(sqrtSk , Vk)
print 'Computation time:', time() - start_time ,"seconds.\n"


#====================== Make a prediction ==============================
dic = testU1[1]
start_time = time()
prediction=dot( Uk[dic['user']-1,:] , dot( Sk , Vk[:,dic['movie']-1] ))
print 'Prediction obtained in ', time() - start_time ,"seconds.\n"
#~ '''



"""
#====================== Study of MAE(k) ================================
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


#Find the best approximation 
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
k = 12  #Let's use k=12 for this part
Sk=s[:k,:k]
Uk=transpose(u.T[:k])  
Vk=v[:k]
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

'''
#====================== Ordinary Least Squares =========================
from multiprocessing import Process, Queue, cpu_count
q = Queue()
W=R>0
'''

'''
#===== Study of MAE with k and lambda : Compute MAE-data.txt ===========
#This part takes ~ 40 hours to compute all the MAE values for k in [1,12] and lambda in [0.05,0.9]
#Output file : MAE-data.txt (included)

def getMAE_numpy(k, lambda_, nbrIter):
  X=ones((Nusers,k), dtype=float)
  Y=ones((k,Nmovies), dtype=float)
  print "Now we set X and Y full of ones.\n"
  remindMAE=["\nl=%f\n"%lambda_]
  
  for iteration in range(nbrIter):
    print "iteration %d."%(iteration+1)
    print "estimating X..."
    for u in xrange(Nusers):
      YWu = dot(Y , diag(W[u]))
      X[u,:] = linalg.solve(  dot( YWu, Y.T) + lambda_*eye(k) , 
                            dot( YWu, (R[u]).T ) ).T
    print "estimating Y..."
    for i in xrange(Nmovies):
      XTWi = dot( X.T , diag(W[:,i]) )
      Y[:,i] = linalg.solve(  dot( XTWi , X ) + lambda_*eye(k) , 
                            dot( XTWi , R[:,i] ) )
      
    print "estimating the MAE with X*Y predictions..."
    MAE=0
    for dic in testU1:
      prediction=dot( X[dic['user']-1,:] , Y[:,dic['movie']-1] )
      MAE += abs( prediction - dic['rating'] )
    MAE /= float(len(testU1))
    print "MAE= %f"%MAE
    remindMAE.append(str(MAE))
    
  q.put(remindMAE)  #if used with multiprocessing
  return remindMAE  #if used without multiprocessing


def solveAxEQUb(A,b):
  u,diags,v= linalg.svd(A, full_matrices=0)
  z = linalg.solve( diag(diags) , dot(u.T,b) )
  return dot( v , z )


def getMAE_perso(k, lambda_, nbrIter):
  X=ones((Nusers,k), dtype=float)
  Y=ones((k,Nmovies), dtype=float)
  print "Now we set X and Y full of ones.\n"
  remindMAE=["\nl=%f\n"%lambda_]
  
  for iteration in range(nbrIter):
    print "iteration %d."%(iteration+1)
    print "estimating X..."
    for u in xrange(Nusers):
      YWu = dot(Y , diag(W[u]))
      A = dot( YWu, Y.T) + lambda_*eye(k)
      b = dot( YWu, (R[u]).T )
      X[u,:] = solveAxEQUb( A , b ).T
    print "estimating Y..."
    for i in xrange(Nmovies):
      XTWi = dot( X.T , diag(W[:,i]) )
      A = dot( XTWi , X ) + lambda_*eye(k)
      b = dot( XTWi , R[:,i] )
      Y[:,i] = solveAxEQUb( A , b )
      
    print "estimating the MAE with X*Y predictions..."
    MAE=0
    for dic in testU1:
      prediction=dot( X[dic['user']-1,:] , Y[:,dic['movie']-1] )
      MAE += abs( prediction - dic['rating'] )
    MAE /= float(len(testU1))
    print "MAE= %f"%MAE
    remindMAE.append(str(MAE))
    
  q.put(remindMAE)  #if used with multiprocessing
  return remindMAE  #if used without multiprocessing


nbrIter=35
lambdaRange = [0.05, 1]
krange = [1,2]

Nproc = cpu_count() - 1  #adapts the algorithm to your number of CPU cores, to keep only one free
print "We will use %d CPU cores on %d."%(Nproc, Nproc+1)

for k in krange:
  print "====================== k = %d ==========================="%k
  fichier = open("MAE-data.txt","a")
  fichier.write( "\nk = %d\n"%k )
  fichier.close()
  
  for lambda_ in arange(lambdaRange[0], lambdaRange[1], 0.05*Nproc):
    fichier = open("MAE-data.txt","a")
    ProcessList = []
    for i in range(cpu_count()):  #sets several processes for different lambdas in parallel
      ProcessList.append( Process(target=getMAE_perso, args=(k, lambda_+0.05*i, nbrIter)) )
      
    for i,P in enumerate(ProcessList):  #starts the processes
      print "lambda = %f"%(lambda_+0.05*i)
      P.start()

    for P in ProcessList:
      remindMAE=q.get()
      print remindMAE
      fichier.write( " ".join(remindMAE))
      P.join()  #Waits for the process' end
  
    fichier.close()
'''

'''
#==================== Plot 3D from MAE-data.txt ========================
print "reading data..."
from mpl_toolkits.mplot3d import *
from matplotlib import cm

fig = plt.figure()
ax = Axes3D(fig)

nbrIter = 35
lstep = 0.05
lmax = 0.9
kmax = 11
krange = [0 for i in xrange(int((kmax-1)*lmax/lstep))]
lrange = [0 for i in xrange(int((kmax-1)*lmax/lstep))]
MAErange = [0 for i in xrange(int((kmax-1)*lmax/lstep))]

fichier = open("MAE-data.txt", "r")
lines = fichier.readlines()
k=0
l=0
for line in lines:
  if line[0]=='l':
    l = float(line[2:])
  if line[0]=='k':
    k = int(line[2:])
  if line[0]==' ' or line[0]=='0':
    MAEvalues  = line.split()
    pos = int(round(l/lstep + (k-2)*lmax/lstep-1,2))
    MAErange[pos] = float(MAEvalues[nbrIter-1])
    krange[pos] = k
    lrange[pos] = l

fichier.close()

ax.plot_trisurf(krange, lrange, MAErange, cmap=cm.hot)
plt.suptitle('MAE at %d iterations'%nbrIter)
plt.title('MAE with predictions based on the product X*Y')
plt.xlabel('k')
plt.xlim(2,kmax)
plt.ylabel('lambda')
plt.ylim(0,lmax)
ax.set_zlim(0.73,0.95)
plt.show()
#~ plt.savefig("MAE-least-squares.png",dpi=72,format='png')
'''

'''
#====================== Compute X and Y ================================
start_time_global = time()

nbrIter = 1
lambda_ = 0.85
k = 2

X=ones((Nusers,k), dtype=float)
Y=ones((k,Nmovies), dtype=float)
print "Now we set X and Y full of ones.\n"
remindMAE=["\nl=%f\n"%lambda_]

for iteration in range(nbrIter):
  start_time = time()
  print "iteration %d."%(iteration+1)
  print "estimating X..."
  for u in xrange(Nusers):
    YWu = dot(Y , diag(W[u]))
    X[u,:] = linalg.solve(  dot( YWu, Y.T) + lambda_*eye(k) , dot( YWu, (R[u]).T ) ).T
  print "estimating Y..."
  for i in xrange(Nmovies):
    XTWi = dot( X.T , diag(W[:,i]) )
    Y[:,i] = linalg.solve(  dot( XTWi , X ) + lambda_*eye(k) , dot( XTWi , R[:,i] ) )
  print 'Computation time:', time() - start_time ,"sec"
  
print 'Computation time in seconds:', time() - start_time_global

print dot(X,Y)


#====================== Make a prediction ==============================
dic = testU1[1]
start_time = time()
prediction=dot( X[dic['user']-1,:] , Y[:,dic['movie']-1] )
print 'Prediction obtained in ', time() - start_time ,"seconds.\n"
'''



#====================== Recommandations ================================
nRec = 20
userID = 944

Recommend = []
rated = []

XuY = dot(X[userID-1,:] , Y)

for rating in baseU1:
  if rating['user'] == userID:
    rated.append(rating['movie'])
for movie in xrange(Nmovies):
  note = XuY[movie]
  if movie not in rated and note>=3:
    Recommend.append( (note,movie) )

Recommend.sort()
Recommend = Recommend[-nRec:]
Recommend.reverse()
print "We recommend to user %d the following movies:"%userID
for movie in Recommend:
  print Items[movie[1]]['name']
