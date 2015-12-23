import re
import math

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

	#increase the count of a feature/category pair
	def incf(self,f,cat):
		self.fc.setdefault(f,{})
		self.fc[f].setdefault(cat,0)
		self.fc[f][cat]+=1

	#increase the count of category
	def incc(self,cat):
		self.cc.setdefault(cat,0)
		self.cc[cat]+=1


	# the number of times a feature has appeared in a category
	def fcount(self,f,cat):
		if f in self.fc and cat in self.fc[f]:
			return float(self.fc[f][cat])
		return 0.0

	#number of times in a category
	def catcount(self,cat):
		if cat in self.cc:
			return float(self.cc[cat])
		return 0

	#total number of items
	def totalcount(self):
		return sum(self.cc.values())

	#list of all categories
	def categories(self):
		return self.cc.keys()

	def train(self,item,cat):
		features =self.getfeatures(item)

		#increment count for every featre with this catgory
		for f in features:
			self.incf(f,cat)

		#increment count for this category
		self.incc(cat)


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