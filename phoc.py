import numpy as np
import more_itertools as mit 
import re

alphabet=[chr(i) for i in range(97,123)]
digits=[chr(i) for i in range(48,58)]
bigrams=[ 'th' , 'he'  ,'in'  ,'er' , 'an'  ,'re'  ,'on'  ,'at'  ,'en'  ,'nd'  , 'ti' ,'es'  ,'or'  ,'te'  ,'of'  ,'ed'  ,'is'  ,'it'  ,'al'   ,'ar'   , 'st'   ,'to' ,'nt'   ,'ng'   ,'se'   ,'ha'   ,'as'   ,'ou'   ,'io'   ,'le'   ,'ve'   ,'co'   ,'me'   ,'de'   ,'hi'   ,'ri'   ,'ro'   ,'ic'   , 'ne'   ,'ea'   ,'ra'   ,'ce'   ,'li'   ,'ch'   ,'ll'   ,'be'   ,'ma'   ,'si'   ,'om'   ,'ur' ]




class develop_phoc():

    #This function splits the string into many substrings as per the required level of split

        
    def split_string(string,level):

            iterable=list(string)
            splits=[''.join(list(c)) for c in mit.divide(level,iterable)]

            return(splits)  


    #This function extacts the features from each substring :  36 dimensional / 86 dimensional(for level 2)

    def extract_feature(sub_string,level):

            substring_feature=[]
            alphabet_feature=[1 if element in sub_string else 0 for element in alphabet]
            digits_feature=[1 if element in sub_string else 0 for element in digits]
            substring_feature=alphabet_feature+digits_feature
        
            #BIGRAM FEATURE ADDED FOR LEVEL 2

            if(level==2):
                bigram_feature=[1 if len(re.findall(b,sub_string))!=0 else 0 for b in bigrams]
                substring_feature=substring_feature+bigram_feature

            return(substring_feature)
            

    #This function combines the above two functions and returns the final PHOC representation

    def phoc_representation(string):

            split_levels=[2,3,4,5]
            feature_vector=[]
        
            for split in split_levels:

                if(split <= len(string)):
                    substrings=develop_phoc.split_string(string,split)
                    for sub in substrings:
                        level_feature=develop_phoc.extract_feature(sub,split)
                        feature_vector=feature_vector+level_feature
                else:
                    level_feature=list(np.zeros(86)) if split==2 else list(np.zeros(36))
                    feature_vector=feature_vector+level_feature        

            return(feature_vector)  






            

































