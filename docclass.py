import re
import math
import sqlite3

def getwords(doc):
	splitter=re.compile('\\W*')
	#Split the words by non aplpha characters
	words = [s.lower() for s in splitter.split(doc) if len(s)>2 and len(s)<20]

	#return the unique set of words only
	return dict([(w,1) for w in words])

#Sample function to train classifier
def sampletrain(c1):
	c1.train('Nobody owns the water','good')
	c1.train('the quick rabbit jumps fances','good')
	c1.train('buy pharmaceuticals now','bad')
	c1.train('make quick money at online casino','bad')
	c1.train('the quick brown fox jumps over the lazy dog','good')

class classifier:
	def __init__(self,getfeatures,filename=None):
		#count the feature/category combinations
		self.fc={}
		#count the documets in each category
		self.cc={}
		self.getfeatures=getfeatures

	def setdb(self,dbfile):
		self.con=sqlite3.connect(dbfile)
		self.con.execute('create table if not exists fc(feature,category,count)')
		self.con.execute('create table if not exists cc(category,count)')

	#increase the count of a feature/category pair
	def incf(self,f,cat):
		#self.fc.setdefault(f,{})
		#self.fc[f].setdefault(cat,0)
		#self.fc[f][cat]+=1
		count=self.fcount(f,cat)
		if count==0:
			self.con.execute("insert into fc values ('%s','%s',1)" %(f,cat))
		else:
			self.con.execute("update fc set count=%d where feature='%s' and category='%s'" %(count+1,f,cat))

	#increase the count of category
	def incc(self,cat):
		#self.cc.setdefault(cat,0)
		#self.cc[cat]+=1
		count=self.catcount(cat)
		if count == 0:
			self.con.execute("insert into cc values('%s',1)" %(cat))
		else:
			self.con.execute("update cc set count=%d where category='%s'" %(count++1,cat))


	# the number of times a feature has appeared in a category
	def fcount(self,f,cat):
		#if f in self.fc and cat in self.fc[f]:
		#	return float(self.fc[f][cat])
		#return 0.0
		res=self.con.execute("select count from fc where feature='%s' and category='%s'" %(f,cat)).fetchone()
		if res == None:
			return 0
		else:
			return float(res[0])

	#number of times in a category
	def catcount(self,cat):
		#if cat in self.cc:
		#	return float(self.cc[cat])
		#return 0
		res=self.con.execute('select count from cc where category="%s"' %(cat)).fetchone()
		if res==None:
			return 0
		else:
			return float(res[0])

	#total number of items
	def totalcount(self):
		#return sum(self.cc.values())
		res=self.con.execute("select sum(count) from cc").fetchone()
		if res==None:
			return 0
		else:
			return res[0]

	#list of all categories
	def categories(self):
		#return self.cc.keys()
		cur=self.con.execute('select category from cc')
		return [d[0] for d in cur]

	def train(self,item,cat):
		features =self.getfeatures(item)

		#increment count for every featre with this catgory
		for f in features:
			self.incf(f,cat)

		#increment count for this category
		self.incc(cat)
		self.con.commit()


	#calculate the probabelity of a feature in the document of that category
	def fprob(self,f,cat):
		if self.catcount(cat) ==0:
			return 0

		#proabability is the total number of times this feature appeared in this category
		#divided by the total number of itemss in this catogory
		return self.fcount(f,cat)/self.catcount(cat)

	#weightedprob method to give initial guess to the probabilities
	#----ap is the assumed probability
	def weightedprob(self,f,cat,prf,weight=1,ap=0.5):
		#calculae the current prob
		basicprob=prf(f,cat)

		#count the number of times this feature has appeared in all the categories
		totals=sum([self.fcount(f,c) for c in self.categories()])

		#calculate the weighted average
		bp=((weight*ap)+(totals*basicprob))/(weight+totals)
		return bp

	
class naivebayes(classifier):
	def __init__(self,getfeatures):
		classifier.__init__(self,getfeatures)
		self.thresholds = {}
	def docprob(self,item,cat):
		features=self.getfeatures(item)

		# multiply probabilitties of all the features
		p=1
		for f in features:
			p*=self.weightedprob(f,cat,self.fprob)
		return p

	def prob(self,item,cat):
		catprob=self.catcount(cat)/self.totalcount()
		docprob=self.docprob(item,cat)
		return docprob*catprob

	def setthreshold(self,cat,t):
		self.thresholds[cat]=t

	def getthreshold(self,cat):
		#return default if not present
		if cat not in self.thresholds:
			return 1.0

		return self.thresholds[cat]

	def classify(self,item,default=None):
		probs = {}
		#find category with highest prob
		maximum= 0.0
		for cat in self.categories():
			probs[cat]=self.prob(item,cat)
			if probs[cat]>maximum:
				maximum=probs[cat]
				best=cat

		#make sure prob exceed threshhold
		for cat in probs:
			if cat == best:
				continue
			if probs[cat]*self.getthreshold(best)>probs[best]:
				return default
		return best

class fisherclassifier(classifier):
	def __init__(self,getfeatures):
		classifier.__init__(self,getfeatures)
		self.minimums={}

	def setminimum(self,cat,minimum):
		self.minimums[cat]=minimum

	def getminimum(self,cat):
		if cat not in self.minimums:
			return 0
		return self.minimums[cat]

	def classify(self,item,default=None):
		#loop through looking for best result
		best = default
		maximum=0.0
		for c in self.categories():
			p=self.fisherprob(item,c)
			#make sure it exceeds mininum
			if p>self.getminimum(c) and p>maximum:
				best = c
				maximum=p
		return best
	def cprob(self,f,cat):
		clf=self.fprob(f,cat)
		if clf==0:
			return 0
		freqsum = sum([self.fprob(f,c) for c in self.categories()])

		p = clf/(freqsum)
		return p

	def fisherprob(self,item,cat):
		p=1
		features=self.getfeatures(item)
		for f in features:
			p*=(self.weightedprob(f,cat,self.cprob))
		#take natural log and multiply by -2 -->> fisher method
		fscore=-2*math.log(p)

		#return inverse chi2 functuon to get probabaility
		return self.invchi2(fscore,len(features)*2)

	def invchi2(self,chi,df):
		m = chi/2.0
		summation=term=math.exp(-m)
		for i in range(1,df//2):
			term*=m/i
			summation+=term
		return min(summation,1.0)