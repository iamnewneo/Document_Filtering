import re
import math

def getwords(doc):
	splitter=re.compile('\\W*')
	#Split the words by non aplpha characters
	words = [s.lower() for s in splitter.split(doc) if len(s)>2 and len(s)<20]

	#return the unique set of words only
	return dict([(w,1) for w in words])

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