import feedparser
import re

def read(feed,classifier):
	#get the feel entries and loop over them
	f=feedparser.parse(feed)
	for entry in f['entries']:
		print("")
		print("--------")
		#print the contents of the entity
		print('Title '+str(entry['title'].encode('utf-8')))
		print('Publisher '+str(entry['publisher'].encode('utf-8')))
		print(str(entry['summary'].encode('utf-8')))

		#combine all the text to create one item for the classifier
		fulltext='%s\n%s\n%s' %(str(entry['title']),str(entry['publisher']),str(entry['summary']))
		# print the best guess at the current cateory
		print('Guess' + str(classifier.classify(fulltext)))

		#ask user for correct category and train on that
		cl=input('Enter category')
		classifier.train(fulltext,cl)