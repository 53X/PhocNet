import numpy as np

alphabet=[chr(i) for i in range(97,123)]
digits=[chr(i) for i in range(48,58)]
bigrams=[ 'th' , 'he'  ,'in'  ,'er' , 'an'  ,'re'  ,'on'  ,'at'  ,'en'  ,'nd'  ,
          'ti' ,'es'  ,'or'  ,'te'  ,'of'  ,'ed'  ,'is'  ,'it'  ,'al'   ,'ar'   ,
          'st'   ,'to' ,'nt'   ,'ng'   ,'se'   ,'ha'   ,'as'   ,'ou'   ,'io'   ,
          'le'   ,'ve'   ,'co'   ,'me'   ,'de'   ,'hi'   ,'ri'   ,'ro'   ,'ic'   ,
          'ne'   ,'ea'   ,'ra'   ,'ce'   ,'li'   ,'ch'   ,'ll'   ,'be'   ,'ma'   ,'si'   ,'om'   ,'ur' ]




	class develop_phoc():

		#This function splits the string into many substrings as per the required level of split

		
		def split_string(string,level):

			beg,split_counter,jump,error=0,1,len(string)//split,len(string)%split
			offset=jump if (jump>error) else error
			end=offset
			splits=[]
			
			while(split_counter<level):
				substring=string[beg:end]
				beg=end
				end=end+offset
				split_counter+=1
				splits.sppend(substring)
			
			splits.append(string[beg:])
			return(splits)	


		#This function extacts the features from each substring :  36 dimensional  

		def extract_feature(sub_string,level):

			substring_feature=[]
			alphabet_feature=[1 if element in sub_string else 0 for element in alphabet]
			digits_feature=[1 if element in sub_string else 0 for element in digits]
			substring_feature=final_feature+alphabet_feature+digits_feature+bigram_feature
		
			#BIGRAM FEATURE YET TO BE ADDED FOR LEVEL 2

			return(substring_feature)
			

		#This function combines the above two functions and returns the final PHOC representation

		def phoc_representation(string):

			split_levels=[2,3,4,5]
			feature_vector=[]
		
			for split in split_levels:
				substrings=develop_phoc.split_string(string,split)
				for sub in sub_strings:
					level_feature=develop_phoc.extract_feature(sub,split)
				    feature_vector=feature_vector+level_feature

			return(np.array(feature_vector))	


































